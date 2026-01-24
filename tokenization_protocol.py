"""
Tokenization Protocol and "Same Number of Tokens" Guarantee.

This module provides:
- Tokenization reports (token ids, per-token decoded strings, token count)
- Pair comparison (same-length check, diff positions, one-token-diff check)
- Report-friendly Markdown export for token-by-token decomposition
- Heuristic suggestions to fix token length mismatches

Designed for minGPT's BPETokenizer (mingpt/bpe.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict

import torch
from mingpt.bpe import BPETokenizer


# Data structures
@dataclass(frozen=True)
class TokenizationReport:
    text: str
    token_ids: List[int]
    token_strs: List[str]  # decoded per-token strings (may include leading spaces)
    seq_len: int
    decoded_roundtrip: str

    def short_preview(self, max_chars: int = 120) -> str:
        s = self.text.replace("\n", "\\n")
        return s if len(s) <= max_chars else s[: max_chars - 3] + "..."


@dataclass(frozen=True)
class PairComparison:
    clean: TokenizationReport
    corrupt: TokenizationReport
    same_length: bool
    diff_positions: List[int]
    diff_count: int

    @property
    def one_token_diff(self) -> bool:
        return self.same_length and self.diff_count == 1


# Core tokenization helpers
def tokenize_2d(bpe: BPETokenizer, text: str, device: Optional[str] = None) -> torch.LongTensor:
    """
    Returns token ids as a 2D tensor of shape (1, T) as BPETokenizer does.
    """
    ids_2d = bpe(text)  # (1, T)
    if device is not None:
        ids_2d = ids_2d.to(device)
    return ids_2d


def tokenize_1d_ids(bpe: BPETokenizer, text: str) -> List[int]:
    """
    Returns token ids as a python list[int] (1D).
    """
    ids = bpe(text)[0].tolist()
    return [int(x) for x in ids]


def decode_token_id(bpe: BPETokenizer, token_id: int) -> str:
    """
    Decode a single token id into its string form.
    """
    t = torch.tensor([token_id], dtype=torch.long)
    return bpe.decode(t)


def decode_tokens_1d(bpe: BPETokenizer, token_ids: Sequence[int]) -> str:
    """
    Decode a sequence of token ids back into a string.
    """
    t = torch.tensor(list(token_ids), dtype=torch.long)
    return bpe.decode(t)


def per_token_strings(bpe: BPETokenizer, token_ids: Sequence[int]) -> List[str]:
    """
    Per-token decoded strings (important for inspecting leading spaces).
    """
    return [decode_token_id(bpe, int(tid)) for tid in token_ids]


def build_report(bpe: BPETokenizer, text: str) -> TokenizationReport:
    """
    Build a complete tokenization report for one text.
    """
    token_ids = tokenize_1d_ids(bpe, text)
    token_strs = per_token_strings(bpe, token_ids)
    decoded = decode_tokens_1d(bpe, token_ids)
    return TokenizationReport(
        text=text,
        token_ids=token_ids,
        token_strs=token_strs,
        seq_len=len(token_ids),
        decoded_roundtrip=decoded,
    )


# Comparison and validations
def diff_positions(a: Sequence[int], b: Sequence[int]) -> List[int]:
    """
    Returns a list of positions where sequences differ.
    If lengths differ, extra positions beyond min length are included as diffs.
    """
    la, lb = len(a), len(b)
    m = min(la, lb)
    diffs = [i for i in range(m) if int(a[i]) != int(b[i])]
    if la != lb:
        diffs.extend(list(range(m, max(la, lb))))
    return diffs


def compare_clean_corrupt(clean: TokenizationReport, corrupt: TokenizationReport) -> PairComparison:
    diffs = diff_positions(clean.token_ids, corrupt.token_ids)
    same_len = (clean.seq_len == corrupt.seq_len)
    return PairComparison(
        clean=clean,
        corrupt=corrupt,
        same_length=same_len,
        diff_positions=diffs,
        diff_count=len(diffs),
    )


def assert_same_length(clean: TokenizationReport, corrupt: TokenizationReport) -> None:
    if clean.seq_len != corrupt.seq_len:
        raise ValueError(
            f"Token length mismatch: clean={clean.seq_len}, corrupt={corrupt.seq_len}.\n"
            f"Clean preview: {clean.short_preview()}\n"
            f"Corrupt preview: {corrupt.short_preview()}"
        )


def assert_one_token_difference(comp: PairComparison) -> None:
    if not comp.same_length:
        raise ValueError(
            f"Cannot check one-token-diff: lengths differ (clean={comp.clean.seq_len}, corrupt={comp.corrupt.seq_len})."
        )
    if comp.diff_count != 1:
        raise ValueError(
            f"Expected exactly 1 differing token position, found {comp.diff_count}: {comp.diff_positions}\n"
            f"Tip: inspect the per-token strings and adjust the text until only one BPE token changes."
        )


def validate_pair(
    bpe: BPETokenizer,
    clean_text: str,
    corrupt_text: str,
    require_same_length: bool = True,
    require_one_token_diff: bool = True,
) -> PairComparison:
    """
    Tokenize both texts, compare, and (optionally) enforce constraints by raising errors.
    """
    clean = build_report(bpe, clean_text)
    corrupt = build_report(bpe, corrupt_text)
    comp = compare_clean_corrupt(clean, corrupt)

    if require_same_length:
        assert_same_length(clean, corrupt)
    if require_one_token_diff:
        assert_one_token_difference(comp)
    return comp


def format_token_list_for_console(rep: TokenizationReport) -> str:
    """
    Console-friendly token list.
    Shows position, token_id, and repr(token_str) to make spaces visible.
    """
    lines = []
    for i, (tid, s) in enumerate(zip(rep.token_ids, rep.token_strs)):
        lines.append(f"{i:02d} | {tid:5d} | {repr(s)}")
    return "\n".join(lines)


def format_pair_diff_markdown(comp: PairComparison) -> str:
    """
    Markdown table: position-wise clean vs corrupt tokens.
    Great for pasting into the report.
    """
    clean = comp.clean
    corrupt = comp.corrupt
    max_len = max(clean.seq_len, corrupt.seq_len)

    header = "| pos | clean_id | clean_tok | corrupt_id | corrupt_tok | diff? |\n|---:|---:|---|---:|---|:---:|\n"
    rows = []
    for i in range(max_len):
        c_id = clean.token_ids[i] if i < clean.seq_len else None
        k_id = corrupt.token_ids[i] if i < corrupt.seq_len else None
        c_tok = clean.token_strs[i] if i < clean.seq_len else ""
        k_tok = corrupt.token_strs[i] if i < corrupt.seq_len else ""
        diff = "OK" if i in comp.diff_positions else ""
        rows.append(
            f"| {i} | {'' if c_id is None else c_id} | {repr(c_tok)} | {'' if k_id is None else k_id} | {repr(k_tok)} | {diff} |"
        )
    return header + "\n".join(rows) + "\n"


def describe_pair(comp: PairComparison) -> str:
    """
    Human-readable summary.
    """
    return (
        "=== Pair summary ===\n"
        f"Clean tokens:   {comp.clean.seq_len}\n"
        f"Corrupt tokens: {comp.corrupt.seq_len}\n"
        f"Same length?    {comp.same_length}\n"
        f"Diff count:     {comp.diff_count}\n"
        f"Diff positions: {comp.diff_positions}\n"
        f"One-token diff? {comp.one_token_diff}\n"
    )


def suggest_fixes(clean: TokenizationReport, corrupt: TokenizationReport) -> List[str]:
    """
    Heuristics to help the user fix length mismatches / multi-token mismatches.
    Not an automatic fixer; it gives actionable suggestions.
    """
    suggestions: List[str] = []

    if clean.seq_len != corrupt.seq_len:
        suggestions.append(
            "Token length mismatch detected. Common causes: whitespace differences, punctuation attachment, "
            "or swapping a word that tokenizes into a different number of BPE tokens."
        )
        suggestions.append(
            "Try keeping punctuation identical (e.g., 'student.' vs 'student .') and keep spaces consistent around the changed word."
        )
        suggestions.append(
            "Proper nouns are often unstable: try swapping to a more common single-token alternative and re-check."
        )

    diffs = diff_positions(clean.token_ids, corrupt.token_ids)
    if clean.seq_len == corrupt.seq_len and len(diffs) != 1:
        suggestions.append(
            f"More than one token differs ({len(diffs)}). You want exactly 1 differing BPE token position."
        )
        suggestions.append(
            "Inspect per-token strings around the diff positions; often a punctuation or whitespace token is also changing."
        )

    suggestions.append(
        "Remember GPT-2 BPE: tokens in the middle often include a leading space. "
        "If you care about the token 'wizard', the actual token is usually ' wizard'."
    )

    return suggestions
