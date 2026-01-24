
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt  
from matplotlib.colors import TwoSlopeNorm  

import torch  


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def _extract_matrix_and_labels(obj: Any) -> Tuple[np.ndarray, Optional[List[str]], Optional[List[str]]]:
    xlabels = None
    ylabels = None

    if torch.is_tensor(obj):
        return _to_numpy(obj), None, None

    if isinstance(obj, dict):
        candidates = ["matrix", "values", "scores", "heatmap", "data", "arr"]
        mat = None
        for k in candidates:
            if k in obj and (torch.is_tensor(obj[k]) or isinstance(obj[k], (list, tuple, np.ndarray))):
                mat = obj[k]
                break

        if mat is None:
            for v in obj.values():
                if torch.is_tensor(v) or isinstance(v, (list, tuple, np.ndarray)):
                    arr = _to_numpy(v)
                    if arr.ndim >= 2:
                        mat = v
                        break

        if mat is None:
            raise ValueError("No pude encontrar una matriz 2D dentro del .pt (dict).")

        M = _to_numpy(mat)

        for tk in ["tokens", "token_strs", "token_labels", "xlabels"]:
            if tk in obj and isinstance(obj[tk], (list, tuple)) and all(isinstance(t, str) for t in obj[tk]):
                xlabels = list(obj[tk])
                break

        for lk in ["layers", "layer_labels", "ylabels"]:
            if lk in obj and isinstance(obj[lk], (list, tuple)):
                if all(isinstance(t, str) for t in obj[lk]):
                    ylabels = list(obj[lk])
                elif all(isinstance(t, (int, np.integer)) for t in obj[lk]):
                    ylabels = [f"L{int(t)}" for t in obj[lk]]
                break

        return M, xlabels, ylabels

    M = _to_numpy(obj)
    if M.ndim < 2:
        raise ValueError(f"The loaded object does not seem to be a 2D matrix. ndim={M.ndim}")
    return M, None, None


def _symmetric_norm(M: np.ndarray, *, eps: float = 1e-12) -> Optional[TwoSlopeNorm]:
    mmin = float(np.nanmin(M))
    mmax = float(np.nanmax(M))
    v = max(abs(mmin), abs(mmax))
    if not np.isfinite(v) or v < eps:
        return None
    return TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=+v)


def plot_heatmap(
    M: np.ndarray,
    *,
    title: str,
    out_path: Path,
    xlabel: str = "Token position",
    ylabel: str = "Layer",
    xlabels: Optional[List[str]] = None,
    ylabels: Optional[List[str]] = None,
    cbar_label: str = "Value",
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if M.ndim > 2:
        M = np.squeeze(M)
    if M.ndim != 2:
        raise ValueError(f"I was waiting for 2D, I got shape={M.shape}")

    norm = _symmetric_norm(M)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.gca()

    im = ax.imshow(M, aspect="auto", interpolation="nearest", norm=norm)
    ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    n_layers, T = M.shape[0], M.shape[1]

    if ylabels is not None and len(ylabels) == n_layers:
        ax.set_yticks(np.arange(n_layers))
        ax.set_yticklabels(ylabels)
    else:
        step = 1 if n_layers <= 16 else max(1, n_layers // 12)
        yt = np.arange(0, n_layers, step)
        ax.set_yticks(yt)
        ax.set_yticklabels([str(int(i)) for i in yt])

    if xlabels is not None and len(xlabels) == T:
        step = 1 if T <= 16 else max(1, T // 12)
        xt = np.arange(0, T, step)
        ax.set_xticks(xt)
        ax.set_xticklabels([xlabels[int(i)] for i in xt], rotation=45, ha="right")
    else:
        step = 1 if T <= 16 else max(1, T // 12)
        xt = np.arange(0, T, step)
        ax.set_xticks(xt)
        ax.set_xticklabels([str(int(i)) for i in xt])

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--artifact_dir",
        type=str,
        default="artifacts/extra3_intrablock",
        help="Folder that contain matrix_post_attn.pt and matrix_post_mlp.pt",
    )
    ap.add_argument(
        "--post_attn",
        type=str,
        default="matrix_post_attn.pt",
        help="Name of the .pt for post_attn (inside artifact_dir)",
    )
    ap.add_argument(
        "--post_mlp",
        type=str,
        default="matrix_post_mlp.pt",
        help="Name of the .pt para post_mlp (inside artifact_dir)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/extra3_intrablock/heatmaps",
        help="Salida para los PNG",
    )
    ap.add_argument(
        "--value_name",
        type=str,
        default="Logit-diff score (or effect)",
        help="Label of the colorbar.",
    )
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    p_attn = artifact_dir / args.post_attn
    p_mlp = artifact_dir / args.post_mlp

    if not p_attn.exists():
        raise FileNotFoundError(f"It does not exist: {p_attn}")
    if not p_mlp.exists():
        raise FileNotFoundError(f"It does not exist: {p_mlp}")

    obj_attn = torch.load(str(p_attn), map_location="cpu")
    obj_mlp = torch.load(str(p_mlp), map_location="cpu")

    M_attn, xlabels_a, ylabels_a = _extract_matrix_and_labels(obj_attn)
    M_mlp,  xlabels_m, ylabels_m = _extract_matrix_and_labels(obj_mlp)

    xlabels = xlabels_a or xlabels_m
    ylabels = ylabels_a or ylabels_m

    if M_attn.shape != M_mlp.shape:
        raise ValueError(f"Shape mismatch: post_attn {M_attn.shape} vs post_mlp {M_mlp.shape}")

    M_delta = M_mlp - M_attn

    plot_heatmap(
        M_attn,
        title="EXTRA 3 — Patch location: post_attn",
        out_path=out_dir / "extra3_post_attn_heatmap.png",
        xlabels=xlabels,
        ylabels=ylabels,
        cbar_label=args.value_name,
    )

    plot_heatmap(
        M_mlp,
        title="EXTRA 3 — Patch location: post_mlp",
        out_path=out_dir / "extra3_post_mlp_heatmap.png",
        xlabels=xlabels,
        ylabels=ylabels,
        cbar_label=args.value_name,
    )

    plot_heatmap(
        M_delta,
        title="EXTRA 3 — Delta: (post_mlp - post_attn)",
        out_path=out_dir / "extra3_post_mlp_minus_post_attn_heatmap.png",
        xlabels=xlabels,
        ylabels=ylabels,
        cbar_label=f"Δ({args.value_name})",
    )

    print(f"Saved heatmaps to: {out_dir}")
    print(f" - {out_dir / 'extra3_post_attn_heatmap.png'}")
    print(f" - {out_dir / 'extra3_post_mlp_heatmap.png'}")
    print(f" - {out_dir / 'extra3_post_mlp_heatmap.png'}")
    print(f" - {out_dir / 'extra3_post_mlp_minus_post_attn_heatmap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
