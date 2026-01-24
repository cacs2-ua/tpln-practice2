"""
Tokenize clean/corrupt prompts, enforce same-length and one-token-diff,
print per-token decomposition, and export a Markdown token table for the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from mingpt.bpe import BPETokenizer

import tokenization_protocol as tp

CLEAN_TEXT = "Juan Antonio watched my neural network learn to juggle bananas; he called it wizard science and demanded espresso"
CORRUPT_TEXT = "Juan Antonio watched my neural network learn to juggle bananas; he called it algorithm science and demanded espresso"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean", type=str, default=CLEAN_TEXT, help="Clean prompt text")
    p.add_argument("--corrupt", type=str, default=CORRUPT_TEXT, help="Corrupted prompt text")
    p.add_argument("--no-require-one-diff", action="store_true", help="Do not require exactly 1 token difference")
    p.add_argument("--out_md", type=str, default="token_table.md", help="Output markdown file for token table")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bpe = BPETokenizer()

    clean_rep = tp.build_report(bpe, args.clean)
    corrupt_rep = tp.build_report(bpe, args.corrupt)
    comp = tp.compare_clean_corrupt(clean_rep, corrupt_rep)

    print(tp.describe_pair(comp))

    print("=== Clean prompt ===")
    print(clean_rep.text)
    print("\n=== Clean tokens (pos | id | repr(token)) ===")
    print(tp.format_token_list_for_console(clean_rep))

    print("\n=== Corrupt prompt ===")
    print(corrupt_rep.text)
    print("\n=== Corrupt tokens (pos | id | repr(token)) ===")
    print(tp.format_token_list_for_console(corrupt_rep))

    require_one = not args.no_require_one_diff
    try:
        _ = tp.validate_pair(
            bpe=bpe,
            clean_text=args.clean,
            corrupt_text=args.corrupt,
            require_same_length=True,
            require_one_token_diff=require_one,
        )
        print("\nValidation passed.")
    except Exception as e:
        print("\nValidation failed:")
        print(e)
        print("\nSuggestions:")
        for s in tp.suggest_fixes(clean_rep, corrupt_rep):
            print("-", s)

    md = tp.format_pair_diff_markdown(comp)
    out_path = Path(args.out_md)
    out_path.write_text(md, encoding="utf-8")
    print(f"\nWrote Markdown token table to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
