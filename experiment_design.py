"""
Section 4: Experimental Design (clean/corrupted prompts + target tokens + hypothesis)

This module provides:
- A structured ExperimentSpec (clean/corrupt prompts, target tokens A/B, hypothesis)
- Validation utilities:
  - clean/corrupt differ by EXACTLY one BPE token
  - same number of tokens
  - target tokens A/B are SINGLE BPE tokens (usually with leading space)
- Convenience: candidate specs + "pick the first valid one" to avoid tokenization surprises.

It depends on tokenization_protocol.py (Section 3), which you already implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from mingpt.bpe import BPETokenizer

import tokenization_protocol as tp


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class ExperimentSpec:
    """
    Section 4 "contract" for the experiment.

    clean_text and corrupt_text:
      - must have same BPE token length
      - must differ by exactly one BPE token

    token_a_str and token_b_str:
      - intended to be single BPE tokens for the next-token prediction
      - recommended to include leading space (e.g., " Paris", " def")
    """
    clean_text: str
    corrupt_text: str
    token_a_str: str
    token_b_str: str
    hypothesis: str


@dataclass(frozen=True)
class ValidatedExperiment:
    spec: ExperimentSpec
    comparison: tp.PairComparison
    changed_position: int
    token_a_id: int
    token_b_id: int


# -----------------------------
# Token helpers (Token A / Token B)
# -----------------------------

def ensure_leading_space(token_str: str) -> str:
    """
    GPT-2 BPE typically encodes mid-sequence words with a leading space.
    This helper makes it harder to forget that, but does NOT guarantee single-token.
    """
    if token_str.startswith(" "):
        return token_str
    return " " + token_str


def single_token_id(bpe: BPETokenizer, token_str: str) -> int:
    """
    Convert a string (e.g., " def") into a SINGLE BPE token id.
    Raises ValueError if the string tokenizes into multiple tokens.
    """
    ids_2d = bpe(token_str)  # (1, T)
    ids = ids_2d[0].tolist()
    if len(ids) != 1:
        raise ValueError(
            f"Target token string must be a single BPE token, but got {len(ids)} tokens for {repr(token_str)}: {ids}"
        )
    return int(ids[0])


# -----------------------------
# Core validation utilities
# -----------------------------

def changed_token_position(comp: tp.PairComparison) -> int:
    """
    Returns the unique position where clean vs corrupt differ.
    Raises if not exactly one differing token.
    """
    tp.assert_one_token_difference(comp)
    return int(comp.diff_positions[0])


def default_hypothesis(changed_pos: int) -> str:
    """
    A report-ready hypothesis for GPT-2 small activation patching heatmaps.
    """
    return (
        f"Hypothesis: The changed token position (position {changed_pos}) should matter most, "
        "so patching activations at this position across early-to-mid layers should strongly restore the clean-consistent continuation. "
        "Middle layers are expected to dominate because they often integrate and route the key conditioning fact/entity forward through the residual stream. "
        "Late layers may also show a secondary effect near the final token position because they directly refine the next-token logits."
    )


def validate_experiment(bpe: BPETokenizer, spec: ExperimentSpec) -> ValidatedExperiment:
    """
    Full Section 4 validation:
    - clean/corrupt must be same length AND differ by exactly one BPE token
    - token A and token B must each be single BPE tokens (usually with leading space)
    """
    comp = tp.validate_pair(
        bpe=bpe,
        clean_text=spec.clean_text,
        corrupt_text=spec.corrupt_text,
        require_same_length=True,
        require_one_token_diff=True,
    )
    pos = changed_token_position(comp)

    token_a_id = single_token_id(bpe, spec.token_a_str)
    token_b_id = single_token_id(bpe, spec.token_b_str)

    return ValidatedExperiment(
        spec=spec,
        comparison=comp,
        changed_position=pos,
        token_a_id=token_a_id,
        token_b_id=token_b_id,
    )


# -----------------------------
# Candidate specs + auto-pick
# -----------------------------

def candidate_experiments() -> List[ExperimentSpec]:
    """
    A small pool of "creative but simple" candidates.
    We DO NOT assume these always pass one-token-diff constraints;
    that's why pick_first_valid_experiment() exists.
    """
    # Note: token strings here include leading spaces on purpose.
    return [
        ExperimentSpec(
            clean_text="In Python, the keyword to define a function is",
            corrupt_text="In JavaScript, the keyword to define a function is",
            token_a_str=" def",
            token_b_str=" function",
            hypothesis="(auto)",
        ),
        ExperimentSpec(
            clean_text="The capital of France is",
            corrupt_text="The capital of Germany is",
            token_a_str=" Paris",
            token_b_str=" Berlin",
            hypothesis="(auto)",
        ),
        ExperimentSpec(
            clean_text="The chemical symbol for water is",
            corrupt_text="The chemical symbol for salt is",
            token_a_str=" H",
            token_b_str=" Na",
            hypothesis="(auto)",
        ),
        ExperimentSpec(
            clean_text="A triangle has three sides. A",
            corrupt_text="A square has three sides. A",
            token_a_str=" triangle",
            token_b_str=" square",
            hypothesis="(auto)",
        ),
    ]


def pick_first_valid_experiment(bpe: BPETokenizer, specs: Optional[List[ExperimentSpec]] = None) -> ValidatedExperiment:
    """
    Tries a list of candidate ExperimentSpec and returns the first one that:
    - differs by exactly one BPE token
    - has equal BPE length
    - has single-token target tokens A/B

    If none works, raises ValueError with a helpful message.
    """
    specs = specs or candidate_experiments()
    errors: List[str] = []

    for i, s in enumerate(specs):
        # If hypothesis was left as "(auto)", fill it after we know the changed position
        try:
            tmp = validate_experiment(bpe, s if s.hypothesis != "(auto)" else s)
            if tmp.spec.hypothesis == "(auto)":
                auto_h = default_hypothesis(tmp.changed_position)
                s2 = ExperimentSpec(
                    clean_text=s.clean_text,
                    corrupt_text=s.corrupt_text,
                    token_a_str=s.token_a_str,
                    token_b_str=s.token_b_str,
                    hypothesis=auto_h,
                )
                tmp = validate_experiment(bpe, s2)
            return tmp
        except Exception as e:
            errors.append(f"[Candidate {i}] {e}")

    raise ValueError(
        "None of the candidate experiments passed the strict Section 4 constraints.\n"
        "This is normal: GPT-2 BPE tokenization can be surprising.\n\n"
        "What to do:\n"
        "1) Provide your own clean/corrupt prompts and re-run the driver with --clean/--corrupt.\n"
        "2) Ensure the two prompts differ by only one BPE token (see token tables).\n"
        "3) Ensure token A/B are single BPE tokens (often with leading spaces).\n\n"
        "Errors from candidates:\n" + "\n".join(errors)
    )


# -----------------------------
# Report-oriented formatting
# -----------------------------

def section4_markdown(valid: ValidatedExperiment) -> str:
    """
    Produces a compact Markdown block you can paste into the PDF report (Section 4).
    Includes prompts, token stats, target tokens, and the hypothesis.
    """
    comp = valid.comparison
    md_table = tp.format_pair_diff_markdown(comp)

    lines = []
    lines.append("## 4) Experimental Design: Clean/Corrupted Pair + Hypothesis\n")
    lines.append("**Clean prompt:**")
    lines.append(f"`{valid.spec.clean_text}`\n")
    lines.append("**Corrupted prompt:**")
    lines.append(f"`{valid.spec.corrupt_text}`\n")

    lines.append(f"**Tokenization constraint:** both prompts have **{comp.clean.seq_len}** BPE tokens and differ at exactly **one** token position: **{valid.changed_position}**.\n")
    lines.append("**Token-by-token comparison (diff highlighted):**\n")
    lines.append(md_table)

    lines.append("**Target competing tokens (next-token prediction at the last position):**")
    lines.append(f"- Token A (clean-consistent): `{valid.spec.token_a_str}`  (token id: {valid.token_a_id})")
    lines.append(f"- Token B (corrupted-consistent): `{valid.spec.token_b_str}`  (token id: {valid.token_b_id})\n")

    lines.append("**Metric used later (matches the handout):**")
    lines.append("`logit(Token B) − logit(Token A)` from the last-position logits.\n")

    lines.append("**Hypothesis:**")
    lines.append(valid.spec.hypothesis + "\n")

    return "\n".join(lines)
