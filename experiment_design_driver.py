"""
What this does:
- Builds/validates an ExperimentSpec:
  - clean/corrupt prompts: same length and exactly 1 token difference
  - target tokens A/B: each must be a single BPE token id
- Prints all evidence needed for the report
- Writes a report-ready Markdown file section4.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mingpt.bpe import BPETokenizer

import tokenization_protocol as tp
import experiment_design as ed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean", type=str, default=None, help="Clean prompt text (optional)")
    p.add_argument("--corrupt", type=str, default=None, help="Corrupted prompt text (optional)")
    p.add_argument("--token_a", type=str, default=None, help="Token A string (optional, usually with leading space)")
    p.add_argument("--token_b", type=str, default=None, help="Token B string (optional, usually with leading space)")
    p.add_argument("--out_md", type=str, default="section4.md", help="Output markdown file for Section 4")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bpe = BPETokenizer()

    if args.clean and args.corrupt and args.token_a and args.token_b:
        spec = ed.ExperimentSpec(
            clean_text=args.clean,
            corrupt_text=args.corrupt,
            token_a_str=args.token_a,
            token_b_str=args.token_b,
            hypothesis="(auto)",  # we will fill after validation
        )
        valid = ed.validate_experiment(bpe, spec)
        # Fill automatic hypothesis (now that we know changed position)
        spec2 = ed.ExperimentSpec(
            clean_text=spec.clean_text,
            corrupt_text=spec.corrupt_text,
            token_a_str=spec.token_a_str,
            token_b_str=spec.token_b_str,
            hypothesis=ed.default_hypothesis(valid.changed_position),
        )
        valid = ed.validate_experiment(bpe, spec2)
    else:
        # Auto-pick the first candidate that satisfies strict constraints
        valid = ed.pick_first_valid_experiment(bpe)

    comp = valid.comparison

    print(tp.describe_pair(comp))
    print(f"Changed token position: {valid.changed_position}")

    print("\n=== Clean prompt ===")
    print(valid.spec.clean_text)
    print("\n=== Corrupted prompt ===")
    print(valid.spec.corrupt_text)

    print("\n=== Token-by-token (clean) ===")
    print(tp.format_token_list_for_console(comp.clean))

    print("\n=== Token-by-token (corrupt) ===")
    print(tp.format_token_list_for_console(comp.corrupt))

    print("\n=== Target tokens ===")
    print(f"Token A (clean-consistent): {repr(valid.spec.token_a_str)} -> id {valid.token_a_id}")
    print(f"Token B (corrupted-consistent): {repr(valid.spec.token_b_str)} -> id {valid.token_b_id}")

    print("\n=== Hypothesis ===")
    print(valid.spec.hypothesis)

    md = ed.section4_markdown(valid)
    out_path = Path(args.out_md)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nWrote Markdown to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
