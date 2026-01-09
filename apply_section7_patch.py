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

    # Prefer inserting after torch imports
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


def replace_gpt_forward_with_section7(src: str) -> str:
    """
    Replace GPT.forward (the one right before GPT.generate) with a version that supports:
      - Section 5: activation recording + clean cache
      - Section 6: last_logits extraction
      - Section 7: activation patching at (patch_layer, patch_position)

    Patching semantics (mandatory):
      - Run corrupted normally up to layer L
      - Compute x = block(x) for layer L
      - Immediately after that, set x[0, pos, :] = clean_activations[L][pos]
      - Continue forward
    """

    new_forward = r'''
    def forward(
        self,
        idx,
        targets=None,
        *,
        record_activations: bool = False,
        cache_activations: bool = False,
        overwrite_cache: bool = False,
        patch_layer: Optional[int] = None,
        patch_position: Optional[int] = None,
    ):
        """
        Forward pass with optional:
          - activation recording (Section 5)
          - clean cache writing (Section 5)
          - last-position logits storage (Section 6)
          - activation patching (Section 7)

        Activation definition (assignment-standard):
          - residual stream output AFTER each transformer block
          - recorded for each token position
          - stored for batch element 0 only
          - deep-copied via detach().clone()

        Patching (Section 7):
          - if (patch_layer, patch_position) is provided,
            replace x[0, patch_position, :] AFTER block output at patch_layer
            with self.clean_activations[patch_layer][patch_position]
          - patch is applied exactly once per run
          - patch runs never overwrite the clean cache
        """
        # If we're caching, we must record
        record_activations = bool(record_activations or cache_activations)

        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)

        # Clear last_activations to avoid stale reads
        self.last_activations = None

        # -----------------------
        # Section 7: patch request validation
        # -----------------------
        wants_patch = (patch_layer is not None) or (patch_position is not None)
        patch_applied = False

        if wants_patch:
            if cache_activations:
                raise RuntimeError("Invalid run: cannot patch while cache_activations=True (clean-cache write mode).")
            if patch_layer is None or patch_position is None:
                raise RuntimeError("Invalid patch request: both patch_layer and patch_position must be provided together.")
            if self.clean_activations is None:
                raise RuntimeError("Cannot patch: self.clean_activations is None. Run a clean pass with cache_activations=True first.")

            n_layer = int(len(self.transformer.h))
            if not (0 <= int(patch_layer) < n_layer):
                raise RuntimeError(f"patch_layer out of range: {patch_layer} (valid: 0..{n_layer-1})")

            # Validate cached seq_len matches current seq_len
            cached_seq_len = None
            if getattr(self, "clean_activation_meta", None) is not None and "seq_len" in self.clean_activation_meta:
                cached_seq_len = int(self.clean_activation_meta["seq_len"])
            else:
                # fallback: infer from structure
                cached_seq_len = int(len(self.clean_activations[0])) if len(self.clean_activations) > 0 else None

            if cached_seq_len is None:
                raise RuntimeError("Clean cache metadata is missing/corrupt; cannot validate seq_len.")

            if int(t) != int(cached_seq_len):
                raise RuntimeError(
                    f"Sequence length mismatch: current t={t} but cached clean seq_len={cached_seq_len}. "
                    "You must re-cache clean activations for this prompt length."
                )

            if not (0 <= int(patch_position) < int(t)):
                raise RuntimeError(f"patch_position out of range: {patch_position} (valid: 0..{t-1})")

        # -----------------------
        # Forward the GPT model
        # -----------------------
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        acts = None
        if record_activations:
            acts = []  # list[layer][pos] -> Tensor(d_model), batch element 0 only

        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x)

            # -----------------------
            # Section 7: apply patch AFTER layer output and BEFORE next layer consumes it
            # -----------------------
            if wants_patch and (layer_idx == int(patch_layer)):
                clean_vec = self.clean_activations[int(patch_layer)][int(patch_position)]
                # Ensure dtype/device match current residual stream
                clean_vec = clean_vec.to(device=x.device, dtype=x.dtype)

                # In-place replacement (batch element 0, single position)
                with torch.no_grad():
                    x[0, int(patch_position), :].copy_(clean_vec)

                patch_applied = True

            # Record activations AFTER possible patch (important for correctness tests)
            if record_activations:
                layer_acts = []
                for p in range(t):
                    layer_acts.append(x[0, p, :].detach().clone())
                acts.append(layer_acts)

        if wants_patch and not patch_applied:
            # Should never happen due to validation, but keep it loud.
            raise RuntimeError("Patch was requested but not applied (unexpected).")

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # -----------------------
        # Section 6: store last-position logits (next-token distribution after prompt)
        # -----------------------
        self.last_logits = logits[:, -1, :].detach().clone()

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # finalize activation storage
        if record_activations:
            self.last_activations = acts

        # -----------------------
        # Section 5: write clean cache ONLY if explicitly requested
        # -----------------------
        if cache_activations:
            if (self.clean_activations is not None) and (not overwrite_cache):
                raise RuntimeError(
                    "Clean activation cache already exists. "
                    "Pass overwrite_cache=True (or call model.clear_clean_activations()) "
                    "to replace it for a new clean prompt."
                )

            self.clean_activations = acts
            self.clean_activation_meta = {
                "seq_len": int(t),
                "n_layer": int(len(self.transformer.h)),
                "d_model": int(x.shape[-1]),
            }

        # Patch debug info (useful for tests / sanity checks)
        self.last_patch = {
            "requested": bool(wants_patch),
            "applied": bool(patch_applied),
            "layer": patch_layer,
            "position": patch_position,
        }

        return logits, loss
'''.strip("\n")

    # Replace ONLY GPT.forward (the one right before GPT.generate)
    pattern = r"\n\s*def forward\([^\)]*\):\n(?:.|\n)*?(?=\n\s*@torch\.no_grad\(\)\n\s*def generate)"
    if not re.search(pattern, src):
        # More general fallback: any def forward(...) before generate
        pattern2 = r"\n\s*def forward\((?:.|\n)*?\):\n(?:.|\n)*?(?=\n\s*@torch\.no_grad\(\)\n\s*def generate)"
        if not re.search(pattern2, src):
            raise RuntimeError("Could not find GPT.forward block to replace (before generate).")
        src, n = re.subn(pattern2, "\n" + new_forward + "\n", src, count=1)
        if n != 1:
            raise RuntimeError("Unexpected number of replacements for forward().")
        return src

    src, n = re.subn(pattern, "\n" + new_forward + "\n", src, count=1)
    if n != 1:
        raise RuntimeError("Unexpected number of replacements for forward().")
    return src


def main() -> None:
    path = pathlib.Path(mingpt.model.__file__).resolve()
    src = path.read_text(encoding="utf-8")

    src = ensure_typing_import(src)
    src = replace_gpt_forward_with_section7(src)

    path.write_text(src, encoding="utf-8")
    print(f"✅ Section 7 patch applied to: {path}")


if __name__ == "__main__":
    main()
