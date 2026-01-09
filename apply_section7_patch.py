from __future__ import annotations

import pathlib
import re

import mingpt.model


def ensure_typing_import(src: str) -> str:
    """
    Ensure we have: from typing import Any, Dict, List, Optional
    Insert it in a stable spot if missing.
    """
    need_line = "from typing import Any, Dict, List, Optional"
    if need_line in src:
        return src

    # Prefer inserting after torch imports (stable in minGPT)
    pattern = r"(import torch\s*\nimport torch\.nn as nn\s*\n)"
    m = re.search(pattern, src)
    if m:
        insert_at = m.end(1)
        return src[:insert_at] + need_line + "\n" + src[insert_at:]

    # Fallback: after import math
    pattern2 = r"(import math\s*\n)"
    m2 = re.search(pattern2, src)
    if not m2:
        raise RuntimeError("Could not find a safe place to insert typing imports.")
    insert_at = m2.end(1)
    return src[:insert_at] + "\n" + need_line + "\n" + src[insert_at:]


def ensure_instrumentation_attributes(src: str) -> str:
    """
    Ensure GPT.__init__ defines:
      - self.clean_activations
      - self.clean_activation_meta
      - self.last_activations
      - self.last_logits
    We inject them right after the parameter-count print (stable anchor).
    Idempotent: if already present, do nothing.
    """
    if ("self.clean_activations" in src and
        "self.clean_activation_meta" in src and
        "self.last_activations" in src and
        "self.last_logits" in src):
        return src

    marker = 'print("number of parameters: %.2fM" % (n_params/1e6,))'
    if marker not in src:
        raise RuntimeError("Could not find the parameter-count print marker in GPT.__init__.")

    inject = (
        marker
        + "\n\n"
        + "        # --- Mechanistic interpretability instrumentation (Sections 5–7) ---\n"
        + "        # Clean cache: list[layer][position] -> Tensor(d_model) for batch element 0\n"
        + "        self.clean_activations: Optional[List[List[torch.Tensor]]] = None\n"
        + "        self.clean_activation_meta: Optional[Dict[str, int]] = None\n"
        + "        # Debug/inspection: last recorded activations (does NOT overwrite clean cache)\n"
        + "        self.last_activations: Optional[List[List[torch.Tensor]]] = None\n"
        + "        # Section 6: logits at final prompt position (next-token distribution), shape (B, vocab_size)\n"
        + "        self.last_logits: Optional[torch.Tensor] = None\n"
    )
    return src.replace(marker, inject)


def ensure_clear_method(src: str) -> str:
    """
    Ensure GPT has:
        def clear_clean_activations(self) -> None
    Insert before forward() if missing.
    """
    if "def clear_clean_activations" in src:
        return src

    m = re.search(r"\n\s*def forward\(", src)
    if not m:
        raise RuntimeError("Could not find forward() to insert clear_clean_activations() before.")

    insert_at = m.start()
    helper = (
        "\n"
        "    def clear_clean_activations(self) -> None:\n"
        "        \"\"\"Clear the stored clean activation cache (Section 5).\"\"\"\n"
        "        self.clean_activations = None\n"
        "        self.clean_activation_meta = None\n"
        "\n"
    )
    return src[:insert_at] + helper + src[insert_at:]


def replace_forward_with_sections_5_7(src: str) -> str:
    """
    Replace GPT.forward with a version that supports:
      - Section 5: record activations and cache clean activations
      - Section 6: store last-position logits in self.last_logits
      - Section 7: patch ONE (layer, position) by replacing corrupted activation with clean activation

    Patching semantics:
      - patch happens AFTER block output for layer L (after x = block(x))
      - patch happens BEFORE next layer consumes x
      - patch only one (layer, position) per run
      - patched runs NEVER overwrite clean cache
    """
    new_forward = r'''
    def forward(
        self,
        idx,
        targets=None,
        *,
        # Section 5: recording/caching
        record_activations: bool = False,
        cache_activations: bool = False,
        overwrite_cache: bool = False,
        # Section 7: patching controls
        apply_patch: bool = False,
        patch_layer: Optional[int] = None,
        patch_position: Optional[int] = None,
    ):
        """
        Forward pass with optional:
          - activation recording (Section 5)
          - clean activation caching (Section 5)
          - last-position logits capture (Section 6)
          - activation patching (Section 7)

        Activation definition for caching/patching:
          - residual stream output AFTER each transformer block
          - per token position
          - batch element 0 only
          - stored via detach().clone()

        Section 7 patch:
          - if apply_patch=True, replace x[0, patch_position, :] AFTER block `patch_layer`
            with clean_activations[patch_layer][patch_position]
        """
        # If we're caching, we must record.
        record_activations = bool(record_activations or cache_activations)

        # Decide whether patching is requested.
        do_patch = bool(apply_patch)
        if do_patch:
            if patch_layer is None or patch_position is None:
                raise ValueError("apply_patch=True requires both patch_layer and patch_position.")
            if cache_activations:
                raise ValueError("Do not set cache_activations=True on a patched run (would risk polluting clean cache).")

        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)

        # Reset per-run debug holders
        self.last_activations = None
        self.last_logits = None

        # Validate patch indices + cache compatibility (fail fast)
        if do_patch:
            if self.clean_activations is None:
                raise RuntimeError("No clean_activations cache found. Run a clean forward with cache_activations=True first.")
            n_layer = len(self.transformer.h)
            if not (0 <= int(patch_layer) < n_layer):
                raise IndexError(f"patch_layer out of range: {patch_layer} (valid: 0..{n_layer-1})")

            clean_seq_len = None
            if self.clean_activation_meta is not None and "seq_len" in self.clean_activation_meta:
                clean_seq_len = int(self.clean_activation_meta["seq_len"])
            else:
                # fallback from structure
                clean_seq_len = len(self.clean_activations[0]) if len(self.clean_activations) > 0 else None

            if clean_seq_len is None:
                raise RuntimeError("Clean cache meta/structure invalid (cannot infer cached seq_len).")

            if int(t) != int(clean_seq_len):
                raise RuntimeError(f"Sequence length mismatch: current t={t}, cached clean seq_len={clean_seq_len}.")

            if not (0 <= int(patch_position) < t):
                raise IndexError(f"patch_position out of range: {patch_position} (valid: 0..{t-1})")

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        acts = None
        if record_activations:
            acts = []  # list[layer][pos] -> Tensor(d_model), batch element 0 only

        patch_applied = False

        # IMPORTANT: patch must happen AFTER a selected block output and BEFORE next block consumes it.
        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x)

            # ---- Section 7 patch (single location) ----
            if do_patch and (int(layer_idx) == int(patch_layer)):
                # Clean vector (d_model,)
                clean_vec = self.clean_activations[int(patch_layer)][int(patch_position)]
                # Ensure device/dtype match current residual stream
                clean_vec = clean_vec.to(device=x.device, dtype=x.dtype)
                # In-place copy into the corrupted residual stream at (batch=0, position=patch_position)
                x[0, int(patch_position), :].copy_(clean_vec)
                patch_applied = True

            # Record activations AFTER patch (so last_activations shows the intervention)
            if record_activations:
                layer_acts = []
                for p in range(t):
                    layer_acts.append(x[0, p, :].detach().clone())
                acts.append(layer_acts)

        if do_patch and not patch_applied:
            raise RuntimeError(
                f"Patch was requested but never applied (patch_layer={patch_layer}). "
                "This should be impossible if indices were validated."
            )

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # ---- Section 6: store last-position logits (next-token distribution) ----
        self.last_logits = logits[:, -1, :].detach().clone()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # finalize activation storage
        if record_activations:
            self.last_activations = acts

        # ---- Section 5: cache activations as CLEAN cache (only when explicitly asked) ----
        if cache_activations:
            if (self.clean_activations is not None) and (not overwrite_cache):
                raise RuntimeError(
                    "Clean activation cache already exists. "
                    "Pass overwrite_cache=True (or call model.clear_clean_activations()) to replace it."
                )
            self.clean_activations = acts
            self.clean_activation_meta = {
                "seq_len": int(t),
                "n_layer": int(len(self.transformer.h)),
                "d_model": int(x.shape[-1]),
            }

        return logits, loss
'''.strip("\n")

    # Replace ONLY the GPT.forward definition up to just before generate()
    pattern = r"\n\s*def forward\([^\)]*\):\n(?:.|\n)*?(?=\n\s*@torch\.no_grad\(\)\n\s*def generate)"
    if not re.search(pattern, src):
        raise RuntimeError("Could not find GPT.forward block to replace (before generate).")

    src, n = re.subn(pattern, "\n" + new_forward + "\n", src, count=1)
    if n != 1:
        raise RuntimeError(f"Unexpected number of replacements for forward(): {n}")
    return src


def main() -> None:
    path = pathlib.Path(mingpt.model.__file__).resolve()
    src = path.read_text(encoding="utf-8")

    src = ensure_typing_import(src)
    src = ensure_instrumentation_attributes(src)
    src = ensure_clear_method(src)
    src = replace_forward_with_sections_5_7(src)

    path.write_text(src, encoding="utf-8")
    print(f"✅ Section 7 patch applied to: {path}")


if __name__ == "__main__":
    main()
