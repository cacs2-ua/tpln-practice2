from __future__ import annotations

from pathlib import Path
import torch

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed

from section10_visualization import (
    decode_prompt_token_labels,
    plot_logit_diff_heatmap,
    save_figure_publication_quality,
)

CLEAN_TEXT   = "Juan Antonio watched my neural network learn to juggle bananas; he called it wizard science and demanded espresso"
CORRUPT_TEXT = "Juan Antonio watched my neural network learn to juggle bananas; he called it algorithm science and demanded espresso"
TOKEN_A = " wizard"
TOKEN_B = " algorithm"

def single_token_id(bpe: BPETokenizer, s: str) -> int:
    ids = bpe(s)[0].tolist()
    if len(ids) != 1:
        raise ValueError(f"{s!r} is not a single BPE token. Got ids={ids}")
    return int(ids[0])

def score_from_last_logits(last_logits_1d: torch.Tensor, *, a_id: int, b_id: int) -> float:
    return float(last_logits_1d[b_id] - last_logits_1d[a_id])

@torch.no_grad()
def main() -> None:
    set_seed(3407)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = GPT.from_pretrained("gpt2").to(device).eval()
    bpe = BPETokenizer()

    idx_clean = bpe(CLEAN_TEXT).to(device)
    idx_corr  = bpe(CORRUPT_TEXT).to(device)
    if idx_clean.shape != idx_corr.shape:
        raise RuntimeError(f"Token length mismatch: clean={idx_clean.shape}, corrupt={idx_corr.shape}")

    T = int(idx_clean.shape[1])
    n_layers = len(model.transformer.h)
    print(f"Seq len T={T}, n_layers={n_layers}")

    a_id = single_token_id(bpe, TOKEN_A)
    b_id = single_token_id(bpe, TOKEN_B)

    # Clean cache (must populate both intra-block banks)
    _ = model(idx_clean, cache_activations=True, overwrite_cache=True)
    score_clean = score_from_last_logits(model.last_logits[0], a_id=a_id, b_id=b_id)

    # Corrupt baseline
    _ = model(idx_corr)
    score_corr = score_from_last_logits(model.last_logits[0], a_id=a_id, b_id=b_id)

    print("\n=== Baselines ===")
    print(f"score_clean = {score_clean:.4f}")
    print(f"score_corr  = {score_corr:.4f}")

    out_dir = Path("artifacts/extra3_intrablock")
    out_dir.mkdir(parents=True, exist_ok=True)

    token_labels = decode_prompt_token_labels(bpe, CLEAN_TEXT)

    mats = {}
    for loc in ["post_attn", "post_mlp"]:
        print(f"\nSweeping patches at location: {loc}")
        mat = torch.empty((n_layers, T), dtype=torch.float32)

        for L in range(n_layers):
            for P in range(T):
                _ = model(idx_corr,
                          layer_to_patch=L,
                          position_to_patch=P,
                          patch_location=loc)
                s = score_from_last_logits(model.last_logits[0], a_id=a_id, b_id=b_id)
                mat[L, P] = float(s)

        mats[loc] = mat.cpu()
        torch.save(mat.cpu(), out_dir / f"matrix_{loc}.pt")
        print("Saved:", (out_dir / f"matrix_{loc}.pt").resolve())

        title = f"EXTRA 3 — {loc}: logit({TOKEN_B!r}) − logit({TOKEN_A!r})"
        fig, ax = plot_logit_diff_heatmap(
            mat,
            token_labels=token_labels,
            metric_title=title,
            show_token_strings=True,
            center_zero=True,
            include_pos_in_label=True,
        )
        save_figure_publication_quality(fig, out_basepath=out_dir / f"heatmap_{loc}", formats=("png", "pdf"), dpi=300)
        print("Saved heatmap figures for:", loc)

    # Key check: matrices should not be identical
    diff = (mats["post_attn"] - mats["post_mlp"]).abs()
    print("\n=== EXTRA 3 matrix difference stats ===")
    print("max |post_attn - post_mlp| =", float(diff.max()))
    print("mean|post_attn - post_mlp| =", float(diff.mean()))
    print("If max is exactly 0.0, your intra-block split is not taking effect.")

    # Optional: show “restoration” deltas relative to corrupted baseline
    # (patched - corrupt): negative means moving toward clean if clean < corrupt for your pair.
    delta_attn = mats["post_attn"] - float(score_corr)
    delta_mlp  = mats["post_mlp"]  - float(score_corr)
    print("\nDelta sanity (patched - corrupt):")
    print("post_attn: min=", float(delta_attn.min()), "max=", float(delta_attn.max()))
    print("post_mlp : min=", float(delta_mlp.min()),  "max=", float(delta_mlp.max()))

if __name__ == "__main__":
    main()
