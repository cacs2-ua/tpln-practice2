from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import torch


@dataclass(frozen=True)
class Condition:
    name: str
    target: Tuple[int, int]          # (L_target, P_target)
    source: Optional[Tuple[int, int]] # None => no patch; else (L_source, P_source)


def single_token_id(bpe, token_str: str) -> int:
    ids = bpe(token_str)[0].tolist()
    if len(ids) != 1:
        raise ValueError(f"{repr(token_str)} is not a single BPE token. Got {len(ids)} ids: {ids}")
    return int(ids[0])


def score_from_last_logits(last_logits_1d: torch.Tensor, token_a_id: int, token_b_id: int) -> float:
    # score = logit(B) - logit(A)
    return float(last_logits_1d[token_b_id] - last_logits_1d[token_a_id])


def normalized_restoration(score_patched: float, score_clean: float, score_corr: float) -> float:
    denom = (score_clean - score_corr)
    if abs(denom) < 1e-12:
        return float("nan")
    return (score_patched - score_corr) / denom


def conditions_for_target(L: int, P: int, n_layers: int, seq_len: int) -> List[Condition]:
    """
    Returns the 5-condition set (baseline + match + wrong-source variants where valid)
    while keeping the patch TARGET fixed at (L,P).
    """
    conds: List[Condition] = []
    conds.append(Condition("no_patch", (L, P), None))
    conds.append(Condition("match", (L, P), (L, P)))

    # WS-pos +/- (same layer, neighbor token)
    if P + 1 < seq_len:
        conds.append(Condition("WS-pos+", (L, P), (L, P + 1)))
    if P - 1 >= 0:
        conds.append(Condition("WS-pos-", (L, P), (L, P - 1)))

    # WS-layer +/- (same position, neighbor layer)
    if L + 1 < n_layers:
        conds.append(Condition("WS-layer+", (L, P), (L + 1, P)))
    if L - 1 >= 0:
        conds.append(Condition("WS-layer-", (L, P), (L - 1, P)))

    return conds


@torch.no_grad()
def run_condition(
    model,
    idx_corr: torch.Tensor,
    cond: Condition,
    token_a_id: int,
    token_b_id: int,
) -> Dict[str, float]:
    """
    Runs ONE condition and returns score + bookkeeping.
    """
    if cond.source is None:
        _ = model(idx_corr)
        score = score_from_last_logits(model.last_logits[0], token_a_id, token_b_id)
        return {
            "score": score,
            "patched": 0.0,
            "L_target": float(cond.target[0]),
            "P_target": float(cond.target[1]),
            "L_source": float("nan"),
            "P_source": float("nan"),
        }

    (Lt, Pt) = cond.target
    (Ls, Ps) = cond.source
    _ = model(
        idx_corr,
        layer_to_patch=Lt,
        position_to_patch=Pt,
        source_layer=Ls,
        source_position=Ps,
    )
    score = score_from_last_logits(model.last_logits[0], token_a_id, token_b_id)
    return {
        "score": score,
        "patched": 1.0,
        "L_target": float(Lt),
        "P_target": float(Pt),
        "L_source": float(Ls),
        "P_source": float(Ps),
    }
