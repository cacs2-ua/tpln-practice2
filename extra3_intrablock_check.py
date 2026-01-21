from __future__ import annotations

import torch
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer

# Same example as the handout (usually stable)
CLEAN_TEXT   = "Michelle Jones was a top-notch student. Michelle"
CORRUPT_TEXT = "Michelle Smith was a top-notch student. Michelle"
TOKEN_A = " Jones"  # clean-consistent
TOKEN_B = " Smith"  # corrupt-consistent

def single_token_id(bpe: BPETokenizer, s: str) -> int:
    ids = bpe(s)[0].tolist()
    if len(ids) != 1:
        raise ValueError(f"{s!r} is not a single BPE token. Got ids={ids}")
    return int(ids[0])

def score_from_last_logits(last_logits_1d: torch.Tensor, *, a_id: int, b_id: int) -> float:
    # score = logit(B) - logit(A)
    return float(last_logits_1d[b_id] - last_logits_1d[a_id])

@torch.no_grad()
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = GPT.from_pretrained("gpt2").to(device).eval()
    bpe = BPETokenizer()

    idx_clean = bpe(CLEAN_TEXT).to(device)
    idx_corr  = bpe(CORRUPT_TEXT).to(device)

    if idx_clean.shape != idx_corr.shape:
        raise RuntimeError(f"Token length mismatch: clean={idx_clean.shape}, corrupt={idx_corr.shape}")

    T = idx_clean.shape[1]
    n_layers = len(model.transformer.h)
    print(f"Seq len T={T}, n_layers={n_layers}")

    a_id = single_token_id(bpe, TOKEN_A)
    b_id = single_token_id(bpe, TOKEN_B)

    # 1) CLEAN run (this must create BOTH intra-block caches)
    _ = model(idx_clean, cache_activations=True, overwrite_cache=True)
    print("Clean caches present?",
          "post_attn:", model.clean_post_attn_activations is not None,
          "| post_mlp:", model.clean_post_mlp_activations is not None)

    # Show cache dimensions (must be 12 x T for GPT-2 small)
    print("clean_post_attn dims:", len(model.clean_post_attn_activations), "x", len(model.clean_post_attn_activations[0]))
    print("clean_post_mlp  dims:", len(model.clean_post_mlp_activations),  "x", len(model.clean_post_mlp_activations[0]))

    score_clean = score_from_last_logits(model.last_logits[0], a_id=a_id, b_id=b_id)

    # 2) CORR baseline (no patch)
    _ = model(idx_corr)
    score_corr = score_from_last_logits(model.last_logits[0], a_id=a_id, b_id=b_id)

    print("\n=== Baselines ===")
    print(f"score_clean = {score_clean:.4f}")
    print(f"score_corr  = {score_corr:.4f}")
    print("Expected (for this pair): score_clean < score_corr (often clean is negative, corrupt is positive).")

    # Choose a “meaningful” target: the changed-token position is usually 1 for this classic prompt,
    # but we’ll just test a couple positions safely.
    test_L = 6
    test_P = 1 if T > 1 else 0

    # 3) Patch at post-attn
    _ = model(idx_corr,
              layer_to_patch=test_L,
              position_to_patch=test_P,
              patch_location="post_attn")
    s_attn = score_from_last_logits(model.last_logits[0], a_id=a_id, b_id=b_id)
    print("\n=== Patch test (post_attn) ===")
    print("last_patch:", model.last_patch, "last_patch_location:", model.last_patch_location)
    print(f"score_post_attn = {s_attn:.4f}  | delta_vs_corr = {s_attn - score_corr:+.4f}")

    # 4) Patch at post-MLP
    _ = model(idx_corr,
              layer_to_patch=test_L,
              position_to_patch=test_P,
              patch_location="post_mlp")
    s_mlp = score_from_last_logits(model.last_logits[0], a_id=a_id, b_id=b_id)
    print("\n=== Patch test (post_mlp) ===")
    print("last_patch:", model.last_patch, "last_patch_location:", model.last_patch_location)
    print(f"score_post_mlp  = {s_mlp:.4f}  | delta_vs_corr = {s_mlp - score_corr:+.4f}")

    # 5) The key EXTRA 3 assertion: the two locations should not be identical everywhere.
    print("\n=== Intra-block location difference (single cell) ===")
    print(f"abs(score_post_attn - score_post_mlp) = {abs(s_attn - s_mlp):.6f}")
    print("If this is exactly 0.0 for many tested cells, your patch_location may not be applied correctly.")

if __name__ == "__main__":
    main()
