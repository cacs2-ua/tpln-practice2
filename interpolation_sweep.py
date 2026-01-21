from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import torch
from torch.nn import functional as F


Coord = Tuple[int, int]  # (layer, position)


@dataclass(frozen=True)
class Baselines:
    clean_score: float
    corrupt_score: float
    token_a_id: int
    token_b_id: int
    seq_len: int
    n_layer: int


@dataclass(frozen=True)
class Curve:
    coord: Coord
    alphas: List[float]
    scores: List[float]
    restorations: List[float]
    alpha50: Optional[float]


def single_token_id(bpe, token_str: str) -> int:
    ids = bpe(token_str)[0].tolist()
    if len(ids) != 1:
        raise ValueError(f"{repr(token_str)} is not a single BPE token. Got {len(ids)} ids: {ids}")
    return int(ids[0])


def score_from_last_logits(last_logits_1d: torch.Tensor, token_a_id: int, token_b_id: int) -> float:
    # score = logit(B) - logit(A)
    return float(last_logits_1d[token_b_id] - last_logits_1d[token_a_id])


def restoration_fraction(score: float, score_corr: float, score_clean: float) -> float:
    denom = (score_clean - score_corr)
    if abs(denom) < 1e-12:
        return float("nan")
    return (score - score_corr) / denom


def estimate_alpha50(alphas: Sequence[float], restorations: Sequence[float]) -> Optional[float]:
    """
    Returns the smallest alpha where R(alpha) >= 0.5 using linear interpolation.
    If never reaches 0.5 (or NaNs), returns None.
    """
    xs = list(alphas)
    ys = list(restorations)

    # Filter NaNs but keep order
    pairs = [(x, y) for x, y in zip(xs, ys) if (y is not None and not math.isnan(y))]
    if len(pairs) < 2:
        return None

    for i in range(1, len(pairs)):
        x0, y0 = pairs[i - 1]
        x1, y1 = pairs[i]
        if y0 >= 0.5:
            return x0
        if (y0 < 0.5) and (y1 >= 0.5) and (x1 != x0):
            t = (0.5 - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)

    return None


@torch.no_grad()
def compute_baselines(
    model,
    bpe,
    clean_text: str,
    corrupt_text: str,
    token_a_str: str,
    token_b_str: str,
    *,
    device: Optional[str] = None,
    overwrite_cache: bool = True,
) -> Baselines:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    idx_clean = bpe(clean_text).to(device)
    idx_corr = bpe(corrupt_text).to(device)

    if idx_clean.shape[1] != idx_corr.shape[1]:
        raise ValueError(
            f"Token length mismatch: clean T={idx_clean.shape[1]} vs corrupt T={idx_corr.shape[1]}. "
            "They must match for activation patching."
        )

    token_a_id = single_token_id(bpe, token_a_str)
    token_b_id = single_token_id(bpe, token_b_str)

    # Clean run: cache activations
    _ = model(idx_clean, cache_activations=True, overwrite_cache=overwrite_cache)
    if model.last_logits is None:
        raise RuntimeError("model.last_logits missing after clean run.")
    clean_score = score_from_last_logits(model.last_logits[0], token_a_id, token_b_id)

    # Corrupt baseline
    _ = model(idx_corr)
    if model.last_logits is None:
        raise RuntimeError("model.last_logits missing after corrupt run.")
    corrupt_score = score_from_last_logits(model.last_logits[0], token_a_id, token_b_id)

    n_layer = len(model.transformer.h)
    seq_len = int(idx_clean.shape[1])

    return Baselines(
        clean_score=clean_score,
        corrupt_score=corrupt_score,
        token_a_id=token_a_id,
        token_b_id=token_b_id,
        seq_len=seq_len,
        n_layer=n_layer,
    )


@torch.no_grad()
def full_patch_matrix(
    model,
    bpe,
    corrupt_text: str,
    baselines: Baselines,
    *,
    device: Optional[str] = None,
) -> torch.Tensor:
    """
    Computes the standard heatmap scores for alpha=1 patching:
      M[L, P] = score after patching (L,P) with clean (L,P).
    Shape: (n_layer, seq_len)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    idx_corr = bpe(corrupt_text).to(device)

    M = torch.empty((baselines.n_layer, baselines.seq_len), dtype=torch.float32)
    for L in range(baselines.n_layer):
        for P in range(baselines.seq_len):
            _ = model(
                idx_corr,
                layer_to_patch=L,
                position_to_patch=P,
                patch_alpha=1.0,
            )
            score = score_from_last_logits(model.last_logits[0], baselines.token_a_id, baselines.token_b_id)
            M[L, P] = score
    return M


def select_hotspots(
    M: torch.Tensor,
    baselines: Baselines,
    *,
    top_k: int = 3,
) -> Tuple[List[Coord], Coord]:
    """
    Picks:
      - hotspots: top_k cells with highest restoration at alpha=1
      - coldspot: cell with minimal |score - corrupt_score| (near-zero effect)
    """
    n_layer, T = M.shape
    R = torch.empty_like(M)
    for L in range(n_layer):
        for P in range(T):
            R[L, P] = float(
                restoration_fraction(float(M[L, P]), baselines.corrupt_score, baselines.clean_score)
            )

    # Flatten + sort by restoration descending (best restoration first)
    flat = []
    for L in range(n_layer):
        for P in range(T):
            r = float(R[L, P])
            if not math.isnan(r):
                flat.append(((L, P), r))

    flat.sort(key=lambda x: x[1], reverse=True)
    hotspots = [coord for coord, _ in flat[:top_k]]

    # Coldspot: closest to corrupt baseline (small absolute effect)
    best_cold = (0, 0)
    best_dist = float("inf")
    for L in range(n_layer):
        for P in range(T):
            dist = abs(float(M[L, P]) - baselines.corrupt_score)
            if dist < best_dist:
                best_dist = dist
                best_cold = (L, P)

    return hotspots, best_cold


@torch.no_grad()
def interpolation_curve(
    model,
    bpe,
    corrupt_text: str,
    baselines: Baselines,
    coord: Coord,
    alphas: Sequence[float],
    *,
    device: Optional[str] = None,
    source_coord: Optional[Coord] = None,  # optional wrong-source + interpolation together
) -> Curve:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    idx_corr = bpe(corrupt_text).to(device)

    L, P = coord
    if source_coord is None:
        srcL, srcP = L, P
    else:
        srcL, srcP = source_coord

    scores: List[float] = []
    restorations: List[float] = []
    alphas_out: List[float] = []

    for a in alphas:
        a = float(a)
        _ = model(
            idx_corr,
            layer_to_patch=L,
            position_to_patch=P,
            source_layer=srcL,
            source_position=srcP,
            patch_alpha=a,
        )
        sc = score_from_last_logits(model.last_logits[0], baselines.token_a_id, baselines.token_b_id)
        r = restoration_fraction(sc, baselines.corrupt_score, baselines.clean_score)

        alphas_out.append(a)
        scores.append(sc)
        restorations.append(r)

    a50 = estimate_alpha50(alphas_out, restorations)

    return Curve(
        coord=coord,
        alphas=alphas_out,
        scores=scores,
        restorations=restorations,
        alpha50=a50,
    )


def plot_restoration_curves(
    curves: Sequence[Curve],
    *,
    out_path: str = "extra2_interpolation_curves.png",
    title: str = "EXTRA 2: Interpolation sweep (normalized restoration R(alpha))",
) -> str:
    import matplotlib.pyplot as plt

    plt.figure()
    for c in curves:
        L, P = c.coord
        plt.plot(c.alphas, c.restorations, marker="o", label=f"(L={L}, P={P})")

    plt.axhline(0.5, linestyle="--")
    plt.xlabel("alpha")
    plt.ylabel("R(alpha)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path
