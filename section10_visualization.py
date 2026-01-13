"""
Section 10) Visualization: Heatmap Generation and Presentation Standards

This module:
- plots a (n_layers, seq_len) matrix as a heatmap using matplotlib (matshow)
- labels axes (x: token positions or decoded token strings; y: layer indices)
- adds colorbar + title
- saves publication-quality figures (PNG + PDF by default)
- optionally saves/loads matrix + metadata to/from disk for reproducibility
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import json
import math

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ArrayLike = Union[torch.Tensor, Any]  # keep flexible (torch preferred)


@dataclass(frozen=True)
class HeatmapMeta:
    metric_title: str
    clean_text: str
    corrupt_text: str
    token_a_str: str
    token_b_str: str
    n_layers: int
    seq_len: int
    token_labels: Optional[List[str]] = None  # per-position decoded tokens (optional)


def _to_2d_cpu_float(matrix: ArrayLike) -> torch.Tensor:
    if isinstance(matrix, torch.Tensor):
        m = matrix.detach().to("cpu")
    else:
        m = torch.tensor(matrix)
    if m.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {tuple(m.shape)}")
    return m.to(dtype=torch.float32)


def _tick_positions(total: int, max_ticks: int = 40, *, max_xticks: Optional[int] = None) -> List[int]:
    """
    Returns a list of tick indices for a length-`total` axis.

    Accepts both:
      - max_ticks (preferred)
      - max_xticks (backwards/alternate name)
    """
    if max_xticks is not None:
        max_ticks = int(max_xticks)

    if total <= 0:
        return []
    if max_ticks <= 0:
        return list(range(total))
    if total <= max_ticks:
        return list(range(total))

    stride = int(math.ceil(total / max_ticks))
    return list(range(0, total, stride))


def decode_prompt_token_labels(bpe, text: str) -> List[str]:
    """
    Returns per-token decoded strings for the FULL prompt.
    This is exactly what you want for x-axis labels (optional).
    """
    ids_1d = bpe(text)[0].tolist()
    labels: List[str] = []
    for tid in ids_1d:
        tok = bpe.decode(torch.tensor([int(tid)], dtype=torch.long))
        labels.append(tok)
    return labels


def plot_logit_diff_heatmap(
    matrix: ArrayLike,
    *,
    token_labels: Optional[Sequence[str]] = None,
    metric_title: str = "Logit difference heatmap",
    xlabel: str = "Token position",
    ylabel: str = "Layer",
    show_token_strings: bool = True,
    max_xticks: int = 40,
    max_token_label_len: int = 18,
    include_pos_in_label: bool = True,
    center_zero: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> Tuple["plt.Figure", "plt.Axes"]:
    """
    Plots the matrix using matshow.

    Expected matrix shape: (n_layers, seq_len)
      - y-axis: layer index 0..n_layers-1
      - x-axis: token positions 0..seq_len-1
    """
    m = _to_2d_cpu_float(matrix)
    n_layers, seq_len = int(m.shape[0]), int(m.shape[1])

    if figsize is None:
        w = max(7.5, min(16.0, 0.35 * seq_len + 5.0))
        h = max(5.0, min(10.0, 0.28 * n_layers + 4.0))
        figsize = (w, h)

    fig, ax = plt.subplots(figsize=figsize)

    norm = None
    if center_zero:
        _vmin = float(m.min()) if vmin is None else float(vmin)
        _vmax = float(m.max()) if vmax is None else float(vmax)
        if _vmin == _vmax:
            _vmin, _vmax = _vmin - 1.0, _vmax + 1.0
        norm = TwoSlopeNorm(vcenter=0.0, vmin=_vmin, vmax=_vmax)

    im = ax.matshow(m.numpy(), norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("logit(Token B) − logit(Token A)", rotation=90)

    ax.set_title(metric_title, pad=18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # ✅ FIX: use the correct kw name (and helper accepts both anyway)
    xt = _tick_positions(seq_len, max_ticks=max_xticks)
    yt = list(range(n_layers))

    ax.set_xticks(xt)
    ax.set_yticks(yt)
    ax.set_yticklabels([str(i) for i in yt])

    if show_token_strings and token_labels is not None and len(token_labels) == seq_len:
        labels_out: List[str] = []
        for i in xt:
            tok = str(token_labels[i])
            tok_short = tok if len(tok) <= max_token_label_len else (tok[: max_token_label_len - 3] + "...")
            labels_out.append(f"{i}:{tok_short}" if include_pos_in_label else tok_short)
        ax.set_xticklabels(labels_out, rotation=90)
    else:
        ax.set_xticklabels([str(i) for i in xt], rotation=0)

    fig.tight_layout()
    return fig, ax


def save_figure_publication_quality(
    fig: "plt.Figure",
    *,
    out_basepath: Union[str, Path],
    formats: Sequence[str] = ("png", "pdf"),
    dpi: int = 300,
    transparent: bool = False,
    close: bool = True,
) -> List[Path]:
    """
    Saves the figure to out_basepath.{fmt} for each fmt.
    - PNG: high dpi raster
    - PDF: vector-friendly for reports
    """
    out_base = Path(out_basepath)
    out_base.parent.mkdir(parents=True, exist_ok=True)

    saved: List[Path] = []
    for fmt in formats:
        p = out_base.with_suffix("." + fmt.lower())
        fig.savefig(p, dpi=dpi, bbox_inches="tight", transparent=transparent)
        saved.append(p)

    if close:
        plt.close(fig)

    return saved


def save_heatmap_artifacts(
    *,
    out_dir: Union[str, Path],
    matrix: ArrayLike,
    meta: HeatmapMeta,
) -> Path:
    """
    Saves:
      - matrix.pt  (torch tensor, CPU float32)
      - meta.json  (json)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = _to_2d_cpu_float(matrix)
    torch.save(m, out_dir / "matrix.pt")

    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(meta), f, ensure_ascii=False, indent=2)

    return out_dir


def load_heatmap_artifacts(out_dir: Union[str, Path]) -> Tuple[torch.Tensor, HeatmapMeta]:
    out_dir = Path(out_dir)
    m = torch.load(out_dir / "matrix.pt", map_location="cpu")

    with (out_dir / "meta.json").open("r", encoding="utf-8") as f:
        d = json.load(f)

    meta = HeatmapMeta(**d)
    return m, meta
