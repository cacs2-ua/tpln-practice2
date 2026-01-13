from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed

from baseline_utils import run_clean_baseline, run_corrupt_baseline

from section10_visualization import (
    HeatmapMeta,
    decode_prompt_token_labels,
    plot_logit_diff_heatmap,
    save_figure_publication_quality,
    save_heatmap_artifacts,
)

DEFAULT_SAVED = "section9_diff_matrix.pt"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--saved", type=str, default=DEFAULT_SAVED, help="Path to section9 saved matrix (.pt)")
    p.add_argument("--show_token_strs", action="store_true", help="Label x-axis with decoded token strings")
    p.add_argument("--also_delta", action="store_true", help="Also save delta heatmap: score(L,P) - corrupt_score")
    return p.parse_args()


def _safe_token_label(tok: str, max_len: int = 18) -> str:
    """
    Make token strings readable in tick labels:
    - show leading space explicitly as '␠'
    - escape newlines
    - truncate long labels
    """
    tok = tok.replace("\n", "\\n")
    if tok.startswith(" "):
        tok = "␠" + tok[1:]
    if len(tok) > max_len:
        tok = tok[: max_len - 3] + "..."
    return tok


def _load_saved_matrix(path: Path) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Loads either:
      - dict with 'matrix' + metadata
      - raw tensor
    Returns (matrix_float32_cpu, meta_dict)
    """
    obj = torch.load(str(path), map_location="cpu")
    meta: Dict[str, Any] = {}

    if isinstance(obj, dict) and "matrix" in obj:
        matrix = obj["matrix"]
        meta = dict(obj)
    elif torch.is_tensor(obj):
        matrix = obj
        meta = {}
    else:
        raise TypeError(f"Unexpected save format in {path}: {type(obj)}")

    if not torch.is_tensor(matrix) or matrix.ndim != 2:
        raise ValueError(f"Expected a 2D tensor under key 'matrix'. Got: {type(matrix)} shape={getattr(matrix, 'shape', None)}")

    return matrix.detach().to(dtype=torch.float32, device="cpu"), meta


def _tick_positions(total: int, max_ticks: int = 40) -> list[int]:
    if total <= 0:
        return []
    if total <= max_ticks:
        return list(range(total))
    stride = int((total + max_ticks - 1) // max_ticks)
    return list(range(0, total, stride))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(3407)
    device = get_device()
    print("Device:", device)

    saved_path = Path(args.saved)
    if not saved_path.exists():
        raise FileNotFoundError(f"Saved matrix not found: {saved_path.resolve()}")

    # -----------------------
    # Load matrix (+ whatever metadata exists)
    # -----------------------
    matrix, saved_meta = _load_saved_matrix(saved_path)
    n_layers, T = int(matrix.shape[0]), int(matrix.shape[1])
    print(f"Loaded {saved_path} with matrix shape {tuple(matrix.shape)}")

    # Defaults (used if file doesn't contain metadata)
    CLEAN_TEXT = str(saved_meta.get("clean_text", "Michelle Jones was a top-notch student. Michelle"))
    CORRUPT_TEXT = str(saved_meta.get("corrupt_text", "Michelle Smith was a top-notch student. Michelle"))
    TOKEN_A = str(saved_meta.get("token_a", " Jones"))
    TOKEN_B = str(saved_meta.get("token_b", " Smith"))

    saved_clean_score = saved_meta.get("clean_score", float("nan"))
    saved_corrupt_score = saved_meta.get("corrupt_score", float("nan"))

    # -----------------------
    # Recompute baselines (robust for report), but keep saved values if you prefer
    # -----------------------
    model = GPT.from_pretrained("gpt2").to(device).eval()
    bpe = BPETokenizer()

    clean_res = run_clean_baseline(
        model,
        bpe,
        clean_text=CLEAN_TEXT,
        token_a_str=TOKEN_A,
        token_b_str=TOKEN_B,
        device=device,
        top_k=20,
        overwrite_cache=True,
    )
    corrupt_res = run_corrupt_baseline(
        model,
        bpe,
        corrupt_text=CORRUPT_TEXT,
        token_a_str=TOKEN_A,
        token_b_str=TOKEN_B,
        device=device,
        top_k=20,
    )

    clean_score = float(clean_res.score_logit_diff)
    corrupt_score = float(corrupt_res.score_logit_diff)

    print("\nBaselines (for report):")
    print(f"clean_score   = {clean_score:.4f}")
    print(f"corrupt_score = {corrupt_score:.4f}")
    print(f"delta (corrupt-clean) = {corrupt_score - clean_score:.4f}")

    if isinstance(saved_meta, dict) and (not (saved_clean_score != saved_clean_score) or not (saved_corrupt_score != saved_corrupt_score)):
        # Note: NaN check via (x != x)
        print("\nSaved baselines (from file, if present):")
        try:
            print(f"saved_clean_score   = {float(saved_clean_score):.4f}")
            print(f"saved_corrupt_score = {float(saved_corrupt_score):.4f}")
        except Exception:
            pass

    # -----------------------
    # Optional: x-axis token labels
    # -----------------------
    token_labels: Optional[list[str]] = None
    if args.show_token_strs:
        labels_raw = decode_prompt_token_labels(bpe, CORRUPT_TEXT)  # labels should match the prompt used for the matrix
        if len(labels_raw) == T:
            token_labels = [_safe_token_label(s, max_len=18) for s in labels_raw]
        else:
            # fallback: still show positions if mismatch
            token_labels = None

    metric_title = f"Logit difference heatmap: logit({repr(TOKEN_B)}) − logit({repr(TOKEN_A)})\nclean={clean_score:.3f}   corrupt={corrupt_score:.3f}"

    meta = HeatmapMeta(
        metric_title=metric_title,
        clean_text=CLEAN_TEXT,
        corrupt_text=CORRUPT_TEXT,
        token_a_str=TOKEN_A,
        token_b_str=TOKEN_B,
        n_layers=n_layers,
        seq_len=T,
        token_labels=token_labels,
    )

    # Save matrix + metadata for reproducibility
    save_heatmap_artifacts(out_dir=out_dir, matrix=matrix, meta=meta)

    # -----------------------
    # Section 10 deliverable: main heatmap (MATSHOW) + centered diverging norm
    # -----------------------
    fig, ax = plot_logit_diff_heatmap(
        matrix,
        token_labels=token_labels,
        metric_title=metric_title,
        show_token_strings=bool(args.show_token_strs),
        max_xticks=40,
        center_zero=True,
        include_pos_in_label=True,
    )

    fig_base = out_dir / "heatmap_logit_diff"
    saved = save_figure_publication_quality(fig, out_basepath=fig_base, formats=("png", "pdf"), dpi=300)
    plt.close(fig)

    # -----------------------
    # Optional: delta heatmap (patched - corrupt) for interpretability/debug
    # -----------------------
    if args.also_delta:
        delta = matrix - float(corrupt_score)
        max_abs = float(delta.abs().max())
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=max_abs) if max_abs > 0 else None

        fig2 = plt.figure(figsize=(max(10, 0.8 * T), 6))
        ax2 = plt.gca()
        im2 = ax2.matshow(delta.cpu().numpy(), norm=norm, aspect="auto")
        plt.colorbar(im2, label="Δ = score(L,P) − corrupt_score")

        ax2.set_title("Delta heatmap: (patched score − corrupt_score)", pad=18)
        ax2.set_xlabel("Token position")
        ax2.set_ylabel("Layer (0=first, 11=last)")

        xt = _tick_positions(T, max_ticks=40)
        ax2.set_xticks(xt)
        if args.show_token_strs and (token_labels is not None) and len(token_labels) == T:
            ax2.set_xticklabels([f"{i}:{token_labels[i]}" for i in xt], rotation=90, fontsize=8)
        else:
            ax2.set_xticklabels([str(i) for i in xt])

        ax2.set_yticks(list(range(n_layers)))
        ax2.set_yticklabels([str(i) for i in range(n_layers)])

        plt.tight_layout()

        fig2_base = out_dir / "heatmap_delta"
        saved_delta = save_figure_publication_quality(fig2, out_basepath=fig2_base, formats=("png", "pdf"), dpi=300)
        plt.close(fig2)
        saved = list(saved) + list(saved_delta)

    print("\nSaved figures:")
    for p in saved:
        print(" -", Path(p).resolve())


if __name__ == "__main__":
    main()
