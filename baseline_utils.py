from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F

DeviceLike = Union[str, torch.device]


@dataclass(frozen=True)
class TopKEntry:
    rank: int
    token_id: int
    token_str: str
    prob: float
    logit: float


@dataclass(frozen=True)
class BaselineResult:
    prompt: str
    seq_len: int
    topk: List[TopKEntry]
    token_a: str
    token_b: str
    token_a_id: int
    token_b_id: int
    logit_a: float
    logit_b: float
    prob_a: float
    prob_b: float
    score_logit_diff: float  # logit(B) - logit(A)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def single_token_id(bpe, token_str: str) -> int:
    ids = bpe(token_str)[0].tolist()
    if len(ids) != 1:
        raise ValueError(
            f"Target token string must map to exactly 1 BPE token. "
            f"Got {len(ids)} tokens for {repr(token_str)}: {ids}"
        )
    return int(ids[0])


def topk_from_last_logits(bpe, last_logits_1d: torch.Tensor, k: int = 20) -> List[TopKEntry]:
    probs = F.softmax(last_logits_1d, dim=-1)
    top_p, top_i = torch.topk(probs, k)

    out: List[TopKEntry] = []
    for r in range(k):
        tid = int(top_i[r])
        tok = bpe.decode(torch.tensor([tid], dtype=torch.long))  # keep on CPU
        out.append(
            TopKEntry(
                rank=r + 1,
                token_id=tid,
                token_str=tok,
                prob=float(top_p[r]),
                logit=float(last_logits_1d[tid]),
            )
        )
    return out


def compute_logit_diff(last_logits_1d: torch.Tensor, token_b_id: int, token_a_id: int) -> Tuple[float, float, float]:
    logit_a = float(last_logits_1d[token_a_id])
    logit_b = float(last_logits_1d[token_b_id])
    return logit_a, logit_b, (logit_b - logit_a)


@torch.no_grad()
def run_clean_baseline(
    model,
    bpe,
    clean_text: str,
    token_a_str: str,
    token_b_str: str,
    *,
    device: Optional[DeviceLike] = None,
    top_k: int = 20,
    overwrite_cache: bool = True,
) -> BaselineResult:
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    idx = bpe(clean_text).to(device)  # (1, T)
    seq_len = int(idx.shape[1])

    _logits, _loss = model(idx, cache_activations=True, overwrite_cache=overwrite_cache)
    if model.last_logits is None:
        raise RuntimeError("model.last_logits was not set by forward() during clean baseline.")
    last = model.last_logits[0]  # (vocab,)

    token_a_id = single_token_id(bpe, token_a_str)
    token_b_id = single_token_id(bpe, token_b_str)
    logit_a, logit_b, score = compute_logit_diff(last, token_b_id=token_b_id, token_a_id=token_a_id)

    probs = F.softmax(last, dim=-1)
    prob_a = float(probs[token_a_id])
    prob_b = float(probs[token_b_id])

    topk = topk_from_last_logits(bpe, last, k=top_k)

    return BaselineResult(
        prompt=clean_text,
        seq_len=seq_len,
        topk=topk,
        token_a=token_a_str,
        token_b=token_b_str,
        token_a_id=token_a_id,
        token_b_id=token_b_id,
        logit_a=logit_a,
        logit_b=logit_b,
        prob_a=prob_a,          
        prob_b=prob_b,          
        score_logit_diff=score,
    )


@torch.no_grad()
def run_corrupt_baseline(
    model,
    bpe,
    corrupt_text: str,
    token_a_str: str,
    token_b_str: str,
    *,
    device: Optional[DeviceLike] = None,
    top_k: int = 20,
) -> BaselineResult:
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    idx = bpe(corrupt_text).to(device)  # (1, T)
    seq_len = int(idx.shape[1])

    _logits, _loss = model(idx)
    if model.last_logits is None:
        raise RuntimeError("model.last_logits was not set by forward() during corrupt baseline.")
    last = model.last_logits[0]  # (vocab,)

    token_a_id = single_token_id(bpe, token_a_str)
    token_b_id = single_token_id(bpe, token_b_str)
    logit_a, logit_b, score = compute_logit_diff(last, token_b_id=token_b_id, token_a_id=token_a_id)

    probs = F.softmax(last, dim=-1)
    prob_a = float(probs[token_a_id])
    prob_b = float(probs[token_b_id])

    topk = topk_from_last_logits(bpe, last, k=top_k)

    return BaselineResult(
        prompt=corrupt_text,     
        seq_len=seq_len,
        topk=topk,
        token_a=token_a_str,
        token_b=token_b_str,
        token_a_id=token_a_id,
        token_b_id=token_b_id,
        logit_a=logit_a,
        logit_b=logit_b,
        prob_a=prob_a,
        prob_b=prob_b,
        score_logit_diff=score,
    )


def format_topk_table(res: BaselineResult, *, max_rows: int = 20) -> str:
    lines = []
    lines.append(f"Prompt (seq_len={res.seq_len}): {res.prompt}")
    lines.append("")
    lines.append("Metric tokens:")
    lines.append(f"  Token A (clean-consistent):   {repr(res.token_a)}  id={res.token_a_id}  logit={res.logit_a:.4f}")
    lines.append(f"  Token B (corrupt-consistent): {repr(res.token_b)}  id={res.token_b_id}  logit={res.logit_b:.4f}")
    lines.append(f"  score = logit(B) - logit(A) = {res.score_logit_diff:.4f}")
    lines.append(f"  P(Token A) = {res.prob_a:.4f}")
    lines.append(f"  P(Token B) = {res.prob_b:.4f}")
    lines.append("")
    lines.append(f"Top-{min(max_rows, len(res.topk))} next-token continuations (by probability):")
    for e in res.topk[:max_rows]:
        lines.append(
            f"{e.rank:02d}. id={e.token_id:5d} tok={repr(e.token_str):>14}  prob={e.prob:.4f}  logit={e.logit:.4f}"
        )
    return "\n".join(lines)
