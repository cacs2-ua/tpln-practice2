from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed

from patching_sweep import build_patching_sweep
from section10_visualization import (
    HeatmapMeta,
    decode_prompt_token_labels,
    plot_logit_diff_heatmap,
    save_figure_publication_quality,
    save_heatmap_artifacts,
)

def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, default="artifacts/section10", help="Where to save matrix + figures")
    p.add_argument("--no_token_strings", action="store_true", help="Use numeric x-axis ticks only")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    set_seed(3407)
    device = get_device()
    print("Device:", device)

    # --- Choose your experiment (must satisfy same token length; ideally 1-token diff) ---
    CLEAN_TEXT   = "Michelle Jones was a top-notch student. Michelle"
    CORRUPT_TEXT = "Michelle Smith was a top-notch student. Michelle"
    TOKEN_A = " Jones"  # clean-consistent
    TOKEN_B = " Smith"  # corrupt-consistent

    # Load model + tokenizer
    model = GPT.from_pretrained("gpt2").to(device).eval()
    bpe = BPETokenizer()

    # Build sweep (Section 9 output)
    res = build_patching_sweep(
        model,
        bpe,
        clean_text=CLEAN_TEXT,
        corrupt_text=CORRUPT_TEXT,
        token_a_str=TOKEN_A,
        token_b_str=TOKEN_B,
        overwrite_cache=True,
        progress=True,
    )

    # Optional x-axis token strings (decoded from CLEAN prompt tokens)
    token_labels = decode_prompt_token_labels(bpe, CLEAN_TEXT)

    metric_title = f"Logit difference heatmap: logit({repr(TOKEN_B)}) − logit({repr(TOKEN_A)})"
    meta = HeatmapMeta(
        metric_title=metric_title,
        clean_text=CLEAN_TEXT,
        corrupt_text=CORRUPT_TEXT,
        token_a_str=TOKEN_A,
        token_b_str=TOKEN_B,
        n_layers=res.n_layers,
        seq_len=res.seq_len,
        token_labels=token_labels,
    )

    # Save artifacts (matrix + meta) for reproducibility
    save_heatmap_artifacts(out_dir=out_dir, matrix=res.matrix, meta=meta)

    # Plot
    fig, ax = plot_logit_diff_heatmap(
        res.matrix,
        token_labels=None if args.no_token_strings else token_labels,
        metric_title=metric_title,
        show_token_strings=(not args.no_token_strings),
        max_xticks=40,
        center_zero=True,
        include_pos_in_label=True,
    )

    # Save publication-quality figures
    fig_base = out_dir / "heatmap_logit_diff"
    saved = save_figure_publication_quality(fig, out_basepath=fig_base, formats=("png", "pdf"), dpi=300)
    print("Saved figures:")
    for p in saved:
        print(" -", p.resolve())

    print("\nBaselines (for report):")
    print(f"clean_score   = {res.clean_score:.4f}")
    print(f"corrupt_score = {res.corrupt_score:.4f}")
    print(f"delta (corrupt-clean) = {res.corrupt_score - res.clean_score:.4f}")


if __name__ == "__main__":
    main()
