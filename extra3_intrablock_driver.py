from __future__ import annotations

from pathlib import Path

import torch

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed

import tokenization_protocol as tp

from extra3_intrablock_sweep import run_extra3
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

    CLEAN_TEXT = "Juan Antonio watched my neural network learn to juggle bananas; he called it wizard science and demanded espresso"
    CORRUPT_TEXT = "Juan Antonio watched my neural network learn to juggle bananas; he called it algorithm science and demanded espresso"
    TOKEN_A = " wizard"  # clean-consistent
    TOKEN_B = " algorithm"  # corrupt-consistent

    # Load model + tokenizer
    model = GPT.from_pretrained("gpt2").to(device).eval()
    bpe = BPETokenizer()

    # Validate token constraints (recommended)
    comp = tp.validate_pair(
        bpe=bpe,
        clean_text=CLEAN_TEXT,
        corrupt_text=CORRUPT_TEXT,
        require_same_length=True,
        require_one_token_diff=True,
    )
    print(tp.describe_pair(comp))
    print("Changed token position:", comp.diff_positions[0])

    # Run EXTRA 3
    res = run_extra3(
        model,
        bpe,
        clean_text=CLEAN_TEXT,
        corrupt_text=CORRUPT_TEXT,
        token_a_str=TOKEN_A,
        token_b_str=TOKEN_B,
        overwrite_cache=True,
        progress=True,
    )

    print("\n=== Baselines ===")
    print(f"clean score   = {res.clean_score:.4f}")
    print(f"corrupt score = {res.corrupt_score:.4f}")
    print(f"delta (corrupt-clean) = {res.corrupt_score - res.clean_score:.4f}")

    out_dir = Path("artifacts/extra3_intrablock")
    out_dir.mkdir(parents=True, exist_ok=True)

    token_labels = decode_prompt_token_labels(bpe, CLEAN_TEXT)

    # Save raw matrices for reproducibility
    torch.save(
        {
            "post_attn": res.post_attn_matrix,
            "post_mlp": res.post_mlp_matrix,
            "clean_score": res.clean_score,
            "corrupt_score": res.corrupt_score,
            "token_a": res.token_a_str,
            "token_b": res.token_b_str,
            "clean_text": res.clean_text,
            "corrupt_text": res.corrupt_text,
            "seq_len": res.seq_len,
            "n_layers": res.n_layers,
        },
        out_dir / "extra3_matrices.pt",
    )
    print("Saved:", (out_dir / "extra3_matrices.pt").resolve())

    title_attn = f"EXTRA 3 — post-attn patching: logit({repr(TOKEN_B)}) − logit({repr(TOKEN_A)})"
    meta_attn = HeatmapMeta(
        metric_title=title_attn,
        clean_text=CLEAN_TEXT,
        corrupt_text=CORRUPT_TEXT,
        token_a_str=TOKEN_A,
        token_b_str=TOKEN_B,
        n_layers=res.n_layers,
        seq_len=res.seq_len,
        token_labels=token_labels,
    )
    save_heatmap_artifacts(out_dir=out_dir / "post_attn", matrix=res.post_attn_matrix, meta=meta_attn)
    fig, _ = plot_logit_diff_heatmap(
        res.post_attn_matrix,
        token_labels=token_labels,
        metric_title=title_attn,
        show_token_strings=True,
        center_zero=True,
        include_pos_in_label=True,
    )
    save_figure_publication_quality(fig, out_basepath=out_dir / "post_attn" / "heatmap_post_attn", formats=("png", "pdf"))

    title_mlp = f"EXTRA 3 — post-MLP patching: logit({repr(TOKEN_B)}) − logit({repr(TOKEN_A)})"
    meta_mlp = HeatmapMeta(
        metric_title=title_mlp,
        clean_text=CLEAN_TEXT,
        corrupt_text=CORRUPT_TEXT,
        token_a_str=TOKEN_A,
        token_b_str=TOKEN_B,
        n_layers=res.n_layers,
        seq_len=res.seq_len,
        token_labels=token_labels,
    )
    save_heatmap_artifacts(out_dir=out_dir / "post_mlp", matrix=res.post_mlp_matrix, meta=meta_mlp)
    fig, _ = plot_logit_diff_heatmap(
        res.post_mlp_matrix,
        token_labels=token_labels,
        metric_title=title_mlp,
        show_token_strings=True,
        center_zero=True,
        include_pos_in_label=True,
    )
    save_figure_publication_quality(fig, out_basepath=out_dir / "post_mlp" / "heatmap_post_mlp", formats=("png", "pdf"))

    print("\nHeatmaps saved under:", out_dir.resolve())


if __name__ == "__main__":
    main()
