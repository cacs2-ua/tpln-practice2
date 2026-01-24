from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


def _infer_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def single_token_id(bpe, token_str: str) -> int:
    ids = bpe(token_str)[0].tolist()
    if len(ids) != 1:
        raise ValueError(
            f"Target token string must map to exactly 1 BPE token. "
            f"Got {len(ids)} tokens for {repr(token_str)}: {ids}"
        )
    return int(ids[0])


def logit_diff_from_last_logits(last_logits_1d: torch.Tensor, *, token_a_id: int, token_b_id: int) -> float:
    a = float(last_logits_1d[token_a_id])
    b = float(last_logits_1d[token_b_id])
    return b - a


@dataclass(frozen=True)
class Extra3Result:
    post_attn_matrix: torch.Tensor  # (n_layers, T) on CPU float32
    post_mlp_matrix: torch.Tensor   # (n_layers, T) on CPU float32
    n_layers: int
    seq_len: int
    clean_score: float
    corrupt_score: float
    token_a_str: str
    token_b_str: str
    token_a_id: int
    token_b_id: int
    clean_text: str
    corrupt_text: str


@torch.no_grad()
def sweep_location_from_ids(
    model,
    idx_corrupt: torch.LongTensor,
    *,
    token_a_id: int,
    token_b_id: int,
    patch_location: str,
    layers: Optional[Sequence[int]] = None,
    positions: Optional[Sequence[int]] = None,
    progress: bool = False,
) -> torch.Tensor:
    if idx_corrupt.ndim != 2 or idx_corrupt.shape[0] != 1:
        raise ValueError(f"Expected idx_corrupt shape (1,T). Got {tuple(idx_corrupt.shape)}")

    device = _infer_device(model)
    idx_corrupt = idx_corrupt.to(device)

    n_layers = len(model.transformer.h)
    T = int(idx_corrupt.shape[1])

    layers = list(range(n_layers)) if layers is None else list(layers)
    positions = list(range(T)) if positions is None else list(positions)

    it = [(L, P) for L in layers for P in positions]
    if progress:
        try:
            from tqdm import tqdm  
            it = tqdm(it, desc=f"sweep({patch_location})", total=len(it))
        except Exception:
            pass

    mat = torch.empty((len(layers), len(positions)), dtype=torch.float32, device="cpu")
    layer_index = {L: i for i, L in enumerate(layers)}
    pos_index = {P: j for j, P in enumerate(positions)}

    for L, P in it:
        _logits, _loss = model(
            idx_corrupt,
            layer_to_patch=int(L),
            position_to_patch=int(P),
            patch_location=patch_location,
        )
        if model.last_logits is None:
            raise RuntimeError("model.last_logits was not set. Ensure forward() stores last_logits.")
        last = model.last_logits[0].detach()
        score = logit_diff_from_last_logits(last, token_a_id=token_a_id, token_b_id=token_b_id)
        mat[layer_index[L], pos_index[P]] = float(score)

    return mat


@torch.no_grad()
def run_extra3(
    model,
    bpe,
    *,
    clean_text: str,
    corrupt_text: str,
    token_a_str: str,
    token_b_str: str,
    overwrite_cache: bool = True,
    progress: bool = True,
) -> Extra3Result:
    device = _infer_device(model)

    idx_clean = bpe(clean_text).to(device)
    idx_corr = bpe(corrupt_text).to(device)

    if idx_clean.shape != idx_corr.shape:
        raise ValueError(
            f"Clean/Corrupt token length mismatch: clean={tuple(idx_clean.shape)}, corrupt={tuple(idx_corr.shape)}. "
            "They MUST have the same number of BPE tokens."
        )

    T = int(idx_clean.shape[1])
    n_layers = len(model.transformer.h)

    token_a_id = single_token_id(bpe, token_a_str)
    token_b_id = single_token_id(bpe, token_b_str)

    # 1) clean run: cache BOTH post_attn and post_mlp
    _ = model(idx_clean, cache_activations=True, overwrite_cache=overwrite_cache)
    if model.last_logits is None:
        raise RuntimeError("model.last_logits missing after clean run.")
    clean_score = logit_diff_from_last_logits(model.last_logits[0], token_a_id=token_a_id, token_b_id=token_b_id)

    # 2) corrupted baseline
    _ = model(idx_corr)
    if model.last_logits is None:
        raise RuntimeError("model.last_logits missing after corrupt run.")
    corrupt_score = logit_diff_from_last_logits(model.last_logits[0], token_a_id=token_a_id, token_b_id=token_b_id)

    # 3) sweeps
    post_attn = sweep_location_from_ids(
        model,
        idx_corr,
        token_a_id=token_a_id,
        token_b_id=token_b_id,
        patch_location="post_attn",
        progress=progress,
    )
    post_mlp = sweep_location_from_ids(
        model,
        idx_corr,
        token_a_id=token_a_id,
        token_b_id=token_b_id,
        patch_location="post_mlp",
        progress=progress,
    )

    return Extra3Result(
        post_attn_matrix=post_attn,
        post_mlp_matrix=post_mlp,
        n_layers=n_layers,
        seq_len=T,
        clean_score=float(clean_score),
        corrupt_score=float(corrupt_score),
        token_a_str=token_a_str,
        token_b_str=token_b_str,
        token_a_id=token_a_id,
        token_b_id=token_b_id,
        clean_text=clean_text,
        corrupt_text=corrupt_text,
    )
