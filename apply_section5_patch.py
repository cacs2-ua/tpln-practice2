from __future__ import annotations

import pathlib
import re

import mingpt.model


def ensure_typing_import(src: str) -> str:
    """
    Ensure we have: from typing import Any, Dict, List, Optional
    We insert it right after the torch imports if not present.
    """
    need_line = "from typing import Any, Dict, List, Optional"
    if need_line in src:
        return src

    # Try inserting after torch imports (stable in minGPT)
    pattern = r"(import torch\s*\nimport torch\.nn as nn\s*\n)"
    m = re.search(pattern, src)
    if not m:
        # Fallback: insert after "import math"
        pattern2 = r"(import math\s*\n)"
        m2 = re.search(pattern2, src)
        if not m2:
            raise RuntimeError("Could not find a safe place to insert typing imports.")
        insert_at = m2.end(1)
        return src[:insert_at] + "\n" + need_line + "\n" + src[insert_at:]

    insert_at = m.end(1)
    return src[:insert_at] + need_line + "\n" + src[insert_at:]


def insert_instrumentation_attributes(src: str) -> str:
    """
    Insert instrumentation attributes into GPT.__init__ (only once).
    We place them right after the parameter count print.
    """
    if "self.clean_activations" in src:
        return src

    marker = 'print("number of parameters: %.2fM" % (n_params/1e6,))'
    if marker not in src:
        raise RuntimeError("Could not find the parameter-count print marker in GPT.__init__.")

    inject = (
        marker
        + "\n\n"
        + "        # --- Mechanistic interpretability instrumentation (Section 5) ---\n"
        + "        # clean cache: list[layer][position] -> Tensor(d_model) for batch element 0\n"
        + "        self.clean_activations: Optional[List[List[torch.Tensor]]] = None\n"
        + "        self.clean_activation_meta: Optional[Dict[str, int]] = None\n"
        + "        # last recorded activations (debug/inspection; does NOT overwrite clean cache)\n"
        + "        self.last_activations: Optional[List[List[torch.Tensor]]] = None\n"
    )
    return src.replace(marker, inject)


def insert_clear_method_if_missing(src: str) -> str:
    """
    Add a small helper method to clear clean cache (only once).
    We insert it right before forward() definition.
    """
    if "def clear_clean_activations" in src:
        return src

    # Insert before the forward definition (original minGPT has `def forward(self, idx, targets=None):`)
    anchor = "    def forward(self, idx, targets=None):"
    if anchor not in src:
        # Maybe forward already patched; insert before `def forward(` anyway
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

    helper = (
        "\n"
        "    def clear_clean_activations(self) -> None:\n"
        "        \"\"\"Clear the stored clean activation cache (Section 5).\"\"\"\n"
        "        self.clean_activations = None\n"
        "        self.clean_activation_meta = None\n"
        "\n"
    )
    return src.replace(anchor, helper + anchor)


def replace_forward_with_instrumented(src: str) -> str:
    """
    Replace GPT.forward with an instrumented version that can record activations:
      - after each transformer block
      - for each token position
    Stored as: list[layer][position] -> Tensor(d_model), for batch element 0
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
    ):
        """
        Forward pass with optional activation recording (Section 5).

        Activation definition (standardized for this assignment):
          - residual stream output AFTER each transformer block
          - recorded for each token position
          - stored for batch element 0 only
          - deep-copied via detach().clone()

        Storage:
          - self.clean_activations: persistent "clean run cache" (read later for patching)
          - self.last_activations: last recorded activations (debug/inspection)
        """
        # If we're caching, we must record
        record_activations = bool(record_activations or cache_activations)

        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)

        # Clear last_activations to avoid stale reads
        self.last_activations = None

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        acts = None
        if record_activations:
            acts = []  # list[layer][pos] -> Tensor(d_model)

        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x)

            if record_activations:
                # store ONLY batch element 0
                layer_acts = []
                for p in range(t):
                    # defensive copy: detach + clone
                    layer_acts.append(x[0, p, :].detach().clone())
                acts.append(layer_acts)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # finalize activation storage
        if record_activations:
            self.last_activations = acts

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
                "d_model": int(logits.shape[-1] if False else x.shape[-1]),  # x is (b,t,d_model) here pre-ln_f? ln_f preserves size
            }

        return logits, loss
'''.strip("\n")

    # Replace ONLY the forward() definition inside GPT, stopping before generate()
    # We match from `def forward(self, idx, targets=None):` up to just before `@torch.no_grad()` of generate.
    pattern = r"\n\s*def forward\(self, idx, targets=None\):\n(?:.|\n)*?(?=\n\s*@torch\.no_grad\(\)\n\s*def generate)"
    if not re.search(pattern, src):
        # If forward signature already changed, match more generally:
        pattern2 = r"\n\s*def forward\([^\)]*\):\n(?:.|\n)*?(?=\n\s*@torch\.no_grad\(\)\n\s*def generate)"
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
    src = insert_instrumentation_attributes(src)
    src = insert_clear_method_if_missing(src)
    src = replace_forward_with_instrumented(src)

    path.write_text(src, encoding="utf-8")
    print(f"✅ Section 5 patch applied to: {path}")


if __name__ == "__main__":
    main()
