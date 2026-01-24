"""
Full Activation Patching Sweep (Layer × Position Difference Matrix)

Goal:
- For each layer L and token position P:
  run corrupted input with a patch at (L,P),
  compute the scalar metric: logit(token_B) - logit(token_A)
  from last-position logits,
  store it in matrix[L, P].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch


# Small helpers
def _infer_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def single_token_id(bpe, token_str: str) -> int:
    """
    Convert token_str into EXACTLY one BPE token id.
    Raises ValueError if it tokenizes into multiple tokens.

    Important: For GPT-2 BPE, mid-sequence tokens often need a leading space, e.g. " wizard".
    """
    ids = bpe(token_str)[0].tolist()
    if len(ids) != 1:
        raise ValueError(
            f"Target token string must map to exactly 1 BPE token. "
            f"Got {len(ids)} tokens for {repr(token_str)}: {ids}"
        )
    return int(ids[0])


def logit_diff_from_last_logits(last_logits_1d: torch.Tensor, *, token_a_id: int, token_b_id: int) -> float:
    """
    last_logits_1d: shape (vocab_size,)
    returns score = logit(B) - logit(A)
    """
    a = float(last_logits_1d[token_a_id])
    b = float(last_logits_1d[token_b_id])
    return b - a


# Outputs
@dataclass(frozen=True)
class SweepResult:
    """
    matrix shape: (n_layers, seq_len) on CPU (float32)
    """
    matrix: torch.Tensor
    n_layers: int
    seq_len: int
    token_a_str: str
    token_b_str: str
    token_a_id: int
    token_b_id: int
    clean_score: float
    corrupt_score: float
    clean_text: str
    corrupt_text: str


# Core sweep (tensor-level) - best for tests
@torch.no_grad()
def sweep_from_ids(
    model,
    idx_corrupt: torch.LongTensor,
    *,
    token_a_id: int,
    token_b_id: int,
    layers: Optional[Sequence[int]] = None,
    positions: Optional[Sequence[int]] = None,
    progress: bool = False,
) -> torch.Tensor:
    """
    Compute the full patching matrix given:
    - model already has clean activations cached (model.clean_activations != None)
    - idx_corrupt is token ids tensor shape (1, T)

    Returns:
    - matrix: torch.Tensor on CPU, shape (n_layers, T), dtype float32
    """
    if getattr(model, "clean_activations", None) is None:
        raise RuntimeError("No clean cache found. Run a clean pass with cache_activations=True first.")

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
            from tqdm import tqdm  # type: ignore
            it = tqdm(it, desc="patching sweep", total=len(it))
        except Exception:
            pass

    mat = torch.empty((len(layers), len(positions)), dtype=torch.float32, device="cpu")

    layer_index = {L: i for i, L in enumerate(layers)}
    pos_index = {P: j for j, P in enumerate(positions)}

    for L, P in it:
        _logits, _loss = model(idx_corrupt, layer_to_patch=int(L), position_to_patch=int(P))
        if model.last_logits is None:
            raise RuntimeError("model.last_logits was not set. Ensure forward() stores last_logits.")
        last = model.last_logits[0].detach()  # (vocab,)
        score = logit_diff_from_last_logits(last, token_a_id=token_a_id, token_b_id=token_b_id)
        mat[layer_index[L], pos_index[P]] = float(score)

    return mat

@torch.no_grad()
def build_patching_sweep(
    model,
    bpe,
    *,
    clean_text: str,
    corrupt_text: str,
    token_a_str: str,
    token_b_str: str,
    overwrite_cache: bool = True,
    progress: bool = True,
) -> SweepResult:
    """
    Full pipeline:
    1) tokenize clean/corrupt and enforce equal seq_len
    2) cache clean activations (clean run)
    3) compute baseline clean score and corrupted score
    4) sweep all (layer, position) patches on corrupted prompt
    5) return SweepResult with matrix shape (n_layers, seq_len)

    Metric:
      score = logit(Token B) - logit(Token A), using last-position logits.
    """
    device = _infer_device(model)

    idx_clean = bpe(clean_text).to(device)      # (1, T)
    idx_corrupt = bpe(corrupt_text).to(device)  # (1, T)

    if idx_clean.shape != idx_corrupt.shape:
        raise ValueError(
            f"Clean/Corrupt token length mismatch: clean={tuple(idx_clean.shape)}, corrupt={tuple(idx_corrupt.shape)}. "
            "You MUST make both prompts have the same number of BPE tokens."
        )

    T = int(idx_clean.shape[1])
    n_layers = len(model.transformer.h)

    # Token ids for metric
    token_a_id = single_token_id(bpe, token_a_str)
    token_b_id = single_token_id(bpe, token_b_str)

    # CLEAN run: cache activations + baseline score
    _logits, _loss = model(idx_clean, cache_activations=True, overwrite_cache=overwrite_cache)
    if model.last_logits is None:
        raise RuntimeError("model.last_logits missing after clean run.")
    clean_last = model.last_logits[0].detach()
    clean_score = logit_diff_from_last_logits(clean_last, token_a_id=token_a_id, token_b_id=token_b_id)

    # CORRUPTED baseline (no patch)
    _logits, _loss = model(idx_corrupt)
    if model.last_logits is None:
        raise RuntimeError("model.last_logits missing after corrupt baseline run.")
    corrupt_last = model.last_logits[0].detach()
    corrupt_score = logit_diff_from_last_logits(corrupt_last, token_a_id=token_a_id, token_b_id=token_b_id)

    # FULL sweep on corrupted ids (requires clean cache)
    matrix = sweep_from_ids(
        model,
        idx_corrupt,
        token_a_id=token_a_id,
        token_b_id=token_b_id,
        layers=list(range(n_layers)),
        positions=list(range(T)),
        progress=progress,
    )

    return SweepResult(
        matrix=matrix,
        n_layers=n_layers,
        seq_len=T,
        token_a_str=token_a_str,
        token_b_str=token_b_str,
        token_a_id=token_a_id,
        token_b_id=token_b_id,
        clean_score=float(clean_score),
        corrupt_score=float(corrupt_score),
        clean_text=clean_text,
        corrupt_text=corrupt_text,
    )
