from __future__ import annotations

from pathlib import Path
import torch

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed

import tokenization_protocol as tp
from patching_sweep import build_patching_sweep
from section10_visualization import decode_prompt_token_labels, plot_logit_diff_heatmap, save_figure_publication_quality

@torch.no_grad()
def main() -> None:
    set_seed(3407)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    CLEAN_TEXT   = "Michelle Jones was a top-notch student. Michelle"
    CORRUPT_TEXT = "Michelle Smith was a top-notch student. Michelle"
    TOKEN_A = " Jones"
    TOKEN_B = " Smith"

    bpe = BPETokenizer()
    _ = tp.validate_pair(bpe=bpe, clean_text=CLEAN_TEXT, corrupt_text=CORRUPT_TEXT,
                         require_same_length=True, require_one_token_diff=True)

    model = GPT.from_pretrained("gpt2").to(device).eval()
    token_labels = decode_prompt_token_labels(bpe, CLEAN_TEXT)

    out_dir = Path("artifacts/extra3_postattn_vs_postmlp")
    out_dir.mkdir(parents=True, exist_ok=True)

    # post-MLP (default)
    res_mlp = build_patching_sweep(
        model, bpe,
        clean_text=CLEAN_TEXT, corrupt_text=CORRUPT_TEXT,
        token_a_str=TOKEN_A, token_b_str=TOKEN_B,
        overwrite_cache=True, progress=True,
        patch_location="post_mlp",
    )

    fig, _ = plot_logit_diff_heatmap(
        res_mlp.matrix, token_labels=token_labels,
        metric_title="EXTRA 3 — post-MLP patching heatmap",
        show_token_strings=True, center_zero=True, include_pos_in_label=True
    )
    save_figure_publication_quality(fig, out_basepath=out_dir / "heatmap_post_mlp", formats=("png","pdf"), dpi=300)

    # post-attn
    # IMPORTANT: rebuild model to avoid any cache confusion, then run again
    model = GPT.from_pretrained("gpt2").to(device).eval()
    res_attn = build_patching_sweep(
        model, bpe,
        clean_text=CLEAN_TEXT, corrupt_text=CORRUPT_TEXT,
        token_a_str=TOKEN_A, token_b_str=TOKEN_B,
        overwrite_cache=True, progress=True,
        patch_location="post_attn",
    )

    fig, _ = plot_logit_diff_heatmap(
        res_attn.matrix, token_labels=token_labels,
        metric_title="EXTRA 3 — post-attn patching heatmap",
        show_token_strings=True, center_zero=True, include_pos_in_label=True
    )
    save_figure_publication_quality(fig, out_basepath=out_dir / "heatmap_post_attn", formats=("png","pdf"), dpi=300)

    # difference (post_mlp - post_attn)
    diff = res_mlp.matrix - res_attn.matrix
    fig, _ = plot_logit_diff_heatmap(
        diff, token_labels=token_labels,
        metric_title="EXTRA 3 — (post-MLP − post-attn) difference heatmap",
        show_token_strings=True, center_zero=True, include_pos_in_label=True
    )
    save_figure_publication_quality(fig, out_basepath=out_dir / "heatmap_diff_mlp_minus_attn", formats=("png","pdf"), dpi=300)

    print("Saved to:", out_dir.resolve())

if __name__ == "__main__":
    main()
