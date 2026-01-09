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


def insert_last_logits_attribute(src: str) -> str:
    """
    Add: self.last_logits: Optional[torch.Tensor] = None
    Prefer inserting next to Section 5 instrumentation attributes if present.
    Idempotent.
    """
    if "self.last_logits" in src:
        return src

    # If Section 5 instrumentation exists, insert after last_activations
    pattern = r"(self\.last_activations:\s*Optional\[List\[List\[torch\.Tensor\]\]\]\s*=\s*None\s*\n)"
    m = re.search(pattern, src)
    if m:
        inject = (
            m.group(1)
            + "        # last-token logits (Section 6): logits at final prompt position (next-token distribution)\n"
            + "        self.last_logits: Optional[torch.Tensor] = None\n"
        )
        return src[:m.start(1)] + inject + src[m.end(1):]

    # Fallback: insert after parameter-count print in __init__
    marker = 'print("number of parameters: %.2fM" % (n_params/1e6,))'
    if marker not in src:
        raise RuntimeError("Could not find a safe marker in GPT.__init__ to insert last_logits attribute.")

    inject = (
        marker
        + "\n\n"
        + "        # --- Mechanistic interpretability instrumentation (Section 6) ---\n"
        + "        # logits at final prompt position: shape (B, vocab_size)\n"
        + "        self.last_logits: Optional[torch.Tensor] = None\n"
    )
    return src.replace(marker, inject)


def insert_last_logits_assignment_in_forward(src: str) -> str:
    """
    After logits = self.lm_head(x), insert:
        self.last_logits = logits[:, -1, :].detach().clone()
    Idempotent.
    """
    if re.search(r"self\.last_logits\s*=\s*logits\[:,\s*-1,\s*:\]\.detach\(\)\.clone\(\)", src):
        return src

    # Find logits computation line inside forward
    pattern = r"(\n(\s*)logits\s*=\s*self\.lm_head\(x\)\s*\n)"
    m = re.search(pattern, src)
    if not m:
        raise RuntimeError("Could not find the line `logits = self.lm_head(x)` in GPT.forward.")

    full_match = m.group(1)
    indent = m.group(2)

    insertion = (
        full_match
        + f"{indent}# --- Section 6: store last-position logits (next-token distribution after the prompt) ---\n"
        + f"{indent}# Shape: (B, vocab_size). We detach+clone to avoid accidental mutation across runs.\n"
        + f"{indent}self.last_logits = logits[:, -1, :].detach().clone()\n"
    )

    return src[:m.start(1)] + insertion + src[m.end(1):]


def main() -> None:
    path = pathlib.Path(mingpt.model.__file__).resolve()
    src = path.read_text(encoding="utf-8")

    src = ensure_typing_import(src)
    src = insert_last_logits_attribute(src)
    src = insert_last_logits_assignment_in_forward(src)

    path.write_text(src, encoding="utf-8")
    print(f"✅ Section 6 patch applied to: {path}")


if __name__ == "__main__":
    main()
