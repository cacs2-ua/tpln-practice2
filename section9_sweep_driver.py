from __future__ import annotations

import torch
from pathlib import Path

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed

from patching_sweep import build_patching_sweep

# --- Section 10: visualization helpers ---
from section10_visualization import (
    HeatmapMeta,
    decode_prompt_token_labels,
    plot_logit_diff_heatmap,
    save_figure_publication_quality,
    save_heatmap_artifacts,
)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main() -> None:
    set_seed(3407)

    device = get_device()
    print("Device:", device)

    # --- Choose your experiment here ---
    CLEAN_TEXT = "Michelle Jones was a top-notch student. Michelle"
    CORRUPT_TEXT = "Michelle Smith was a top-notch student. Michelle"

    # IMPORTANT: GPT-2 BPE typically needs leading spaces for mid-sequence words
    TOKEN_A = " Jones"  # clean-consistent
    TOKEN_B = " Smith"  # corrupt-consistent

    # Load model + tokenizer
    model = GPT.from_pretrained("gpt2").to(device).eval()
    bpe = BPETokenizer()

    # Build the sweep matrix (Section 9)
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

    print("\n=== Baseline metric sanity ===")
    print(f"clean score   = {res.clean_score:.4f}")
    print(f"corrupt score = {res.corrupt_score:.4f}")
    print(f"delta (corrupt - clean) = {res.corrupt_score - res.clean_score:.4f}")

    print("\n=== Matrix ===")
    print("shape:", tuple(res.matrix.shape), "(n_layers, seq_len)")
    print("n_layers:", res.n_layers, "seq_len:", res.seq_len)
    print("dtype:", res.matrix.dtype, "device:", res.matrix.device)

    # Save for Section 10 plotting later
    torch.save(
        {
            "matrix": res.matrix,
            "n_layers": res.n_layers,
            "seq_len": res.seq_len,
            "token_a": res.token_a_str,
            "token_b": res.token_b_str,
            "token_a_id": res.token_a_id,
            "token_b_id": res.token_b_id,
            "clean_score": res.clean_score,
            "corrupt_score": res.corrupt_score,
            "clean_text": res.clean_text,
            "corrupt_text": res.corrupt_text,
        },
        "section9_diff_matrix.pt",
    )
    print("\nSaved: section9_diff_matrix.pt")

    # >>> INSERTA AQUÍ EL BLOQUE DE SECTION 10 (al final de main) <<<
    out_dir = Path("artifacts/section9_and_10")
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

    save_heatmap_artifacts(out_dir=out_dir, matrix=res.matrix, meta=meta)

    fig, ax = plot_logit_diff_heatmap(
        res.matrix,
        token_labels=token_labels,
        metric_title=metric_title,
        show_token_strings=True,
        center_zero=True,
        include_pos_in_label=True,
    )

    save_figure_publication_quality(
        fig,
        out_basepath=out_dir / "heatmap_logit_diff",
        formats=("png", "pdf"),
        dpi=300,
    )
    print("Saved Section 10 heatmap to:", out_dir.resolve())


if __name__ == "__main__":
    main()
