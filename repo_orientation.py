"""
What this script does:
- Prints where mingpt is installed.
- Locates mingpt/model.py.
- Extracts/prints the key lines of GPT.forward that matter for the assignment:
  embeddings -> transformer blocks -> ln_f -> lm_head -> logits
- Provides programmatic checks used by unit tests.
"""

from __future__ import annotations

import inspect
import pathlib
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import mingpt
import mingpt.model
from mingpt.model import GPT


@dataclass(frozen=True)
class ForwardLandmarks:
    has_tok_emb: bool
    has_pos_emb: bool
    has_blocks_loop: bool
    has_ln_f: bool
    has_lm_head: bool


def get_paths() -> Dict[str, str]:
    pkg_path = pathlib.Path(mingpt.__file__).resolve()
    model_path = pathlib.Path(mingpt.model.__file__).resolve()
    return {
        "mingpt.__file__": str(pkg_path),
        "mingpt.model.__file__": str(model_path),
    }


def read_model_source() -> str:
    model_path = pathlib.Path(mingpt.model.__file__).resolve()
    return model_path.read_text(encoding="utf-8")


def attn_bias_fix_present(model_source: str) -> bool:
    # Required fix: assert len(keys) == len([k for k in sd if not k.endswith(".attn.bias")])
    return 'len([k for k in sd if not k.endswith(".attn.bias")])' in model_source


def forward_source() -> str:
    return inspect.getsource(GPT.forward)


def find_forward_landmarks(src: str) -> ForwardLandmarks:
    has_tok_emb = ("tok_emb" in src) and ("wte" in src)
    has_pos_emb = ("pos_emb" in src) and ("wpe" in src)

    has_blocks_loop = (
        re.search(r"\bfor\b\s+.+\s+\bin\b\s+.*self\.transformer\.h", src) is not None
        or re.search(r"\bfor\b\s+.+\s+\bin\b\s+.*self\.transformer\['h'\]", src) is not None
    )

    has_ln_f = "ln_f" in src
    has_lm_head = ("lm_head" in src) and ("logits" in src)

    return ForwardLandmarks(
        has_tok_emb=has_tok_emb,
        has_pos_emb=has_pos_emb,
        has_blocks_loop=has_blocks_loop,
        has_ln_f=has_ln_f,
        has_lm_head=has_lm_head,
    )


def print_forward_snippet(src: str, max_lines: int = 80) -> None:
    lines = src.splitlines()
    print("=== GPT.forward (snippet) ===")
    for i, line in enumerate(lines[:max_lines], start=1):
        print(f"{i:03d}: {line}")
    if len(lines) > max_lines:
        print(f"... ({len(lines)-max_lines} more lines)")


def main() -> None:
    paths = get_paths()
    print("=== Installed paths ===")
    for k, v in paths.items():
        print(f"{k}: {v}")

    model_src = read_model_source()
    print("\n=== .attn.bias fix present? ===")
    print(attn_bias_fix_present(model_src))

    fwd_src = forward_source()
    landmarks = find_forward_landmarks(fwd_src)
    print("\n=== Forward pipeline landmarks ===")
    print(landmarks)

    print()
    print_forward_snippet(fwd_src)


if __name__ == "__main__":
    main()
