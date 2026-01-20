from __future__ import annotations

from typing import List, Tuple

import torch

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed

import wrong_source_control as wsc


# -------------------------
# EDIT THESE (your experiment)
# -------------------------
CLEAN_TEXT = "Michelle Jones was a top-notch student. Michelle"
CORRUPT_TEXT = "Michelle Smith was a top-notch student. Michelle"
TOKEN_A_STR = " Jones"   # clean-consistent
TOKEN_B_STR = " Smith"   # corrupt-consistent
TOP_K_HOTSPOTS = 3


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def tokens_same_length(bpe: BPETokenizer, a: str, b: str) -> bool:
    return bpe(a).shape[1] == bpe(b).shape[1]


@torch.no_grad()
def compute_score(model, idx, token_a_id: int, token_b_id: int) -> float:
    _ = model(idx)
    return wsc.score_from_last_logits(model.last_logits[0], token_a_id, token_b_id)


@torch.no_grad()
def compute_match_heatmap(
    model,
    idx_corr: torch.Tensor,
    token_a_id: int,
    token_b_id: int,
) -> torch.Tensor:
    n_layers = len(model.transformer.h)
    seq_len = idx_corr.shape[1]
    out = torch.empty((n_layers, seq_len), dtype=torch.float32)

    for L in range(n_layers):
        for P in range(seq_len):
            _ = model(idx_corr, layer_to_patch=L, position_to_patch=P)  # source defaults to match
            out[L, P] = wsc.score_from_last_logits(model.last_logits[0], token_a_id, token_b_id)
    return out


def select_hotspots_and_cold(
    match_heatmap: torch.Tensor,
    score_clean: float,
    score_corr: float,
    top_k: int = 3,
) -> Tuple[List[Tuple[int, int]], Tuple[int, int]]:
    """
    Hotspots: largest normalized restoration R
    Cold: smallest absolute change vs corrupted baseline
    """
    n_layers, seq_len = match_heatmap.shape
    R = torch.empty_like(match_heatmap)

    denom = (score_clean - score_corr)
    if abs(denom) < 1e-12:
        # degenerate; just pick arbitrary cells
        hot = [(0, 0)]
        cold = (0, 0)
        return hot, cold

    R = (match_heatmap - score_corr) / denom

    # flatten
    flat_R = R.flatten()
    top_vals, top_idx = torch.topk(flat_R, k=min(top_k, flat_R.numel()))
    hotspots = []
    used = set()
    for idx in top_idx.tolist():
        L = idx // seq_len
        P = idx % seq_len
        if (L, P) not in used:
            hotspots.append((L, P))
            used.add((L, P))
        if len(hotspots) >= top_k:
            break

    # cold: minimal |score_patch - score_corr| among remaining cells
    delta = (match_heatmap - score_corr).abs()
    delta_flat = delta.flatten()
    # mask out hotspots
    mask = torch.ones_like(delta_flat, dtype=torch.bool)
    for (L, P) in hotspots:
        mask[L * seq_len + P] = False
    masked_delta = delta_flat.clone()
    masked_delta[~mask] = float("inf")
    cold_idx = int(torch.argmin(masked_delta).item())
    cold = (cold_idx // seq_len, cold_idx % seq_len)

    return hotspots, cold


@torch.no_grad()
def main() -> None:
    set_seed(3407)
    device = get_device()
    print("Device:", device)

    # 1) Load model + tokenizer
    model = GPT.from_pretrained("gpt2").to(device).eval()
    bpe = BPETokenizer()

    # 2) Validate same token length
    if not tokens_same_length(bpe, CLEAN_TEXT, CORRUPT_TEXT):
        raise RuntimeError(
            "CLEAN_TEXT and CORRUPT_TEXT do NOT have the same number of BPE tokens.\n"
            "Fix the texts until they tokenize to the same length."
        )

    # 3) Tokenize prompts
    idx_clean = bpe(CLEAN_TEXT).to(device)     # (1, T)
    idx_corr = bpe(CORRUPT_TEXT).to(device)    # (1, T)
    seq_len = idx_corr.shape[1]
    n_layers = len(model.transformer.h)

    # 4) Token ids for metric
    token_a_id = wsc.single_token_id(bpe, TOKEN_A_STR)
    token_b_id = wsc.single_token_id(bpe, TOKEN_B_STR)

    # 5) Clean baseline (cache activations)
    _ = model(idx_clean, cache_activations=True, overwrite_cache=True)
    score_clean = wsc.score_from_last_logits(model.last_logits[0], token_a_id, token_b_id)

    # 6) Corrupted baseline
    _ = model(idx_corr)
    score_corr = wsc.score_from_last_logits(model.last_logits[0], token_a_id, token_b_id)

    print("\n=== Baselines ===")
    print(f"seq_len={seq_len}, n_layers={n_layers}")
    print(f"score_clean = {score_clean:.6f}")
    print(f"score_corr  = {score_corr:.6f}")

    # 7) Compute MATCH heatmap (standard patch)
    print("\nComputing match heatmap (this is the same sweep as your main analysis)...")
    match_heatmap = compute_match_heatmap(model, idx_corr, token_a_id, token_b_id)
    torch.save(match_heatmap.cpu(), "match_heatmap.pt")
    print("Saved: match_heatmap.pt")

    # 8) Pick top hotspots + one cold cell
    hotspots, cold = select_hotspots_and_cold(match_heatmap, score_clean, score_corr, top_k=TOP_K_HOTSPOTS)
    targets = hotspots + [cold]

    print("\n=== Selected targets ===")
    for i, (L, P) in enumerate(targets):
        s = float(match_heatmap[L, P])
        R = wsc.normalized_restoration(s, score_clean, score_corr)
        tag = "COLD" if (L, P) == cold else "HOT"
        print(f"{i+1:02d}. ({L},{P})  match_score={s:.6f}  R_match={R:.4f}  [{tag}]")

    # 9) Run wrong-source conditions per target
    print("\n=== WRONG-SOURCE CONTROL RESULTS ===")
    print("(Metric: score = logit(B) - logit(A); higher/lower direction depends on your pair)\n")

    for (L, P) in targets:
        conds = wsc.conditions_for_target(L, P, n_layers=n_layers, seq_len=seq_len)

        print(f"\n--- Target (L={L}, P={P}) ---")
        print(f"{'condition':12s} | {'source':10s} | {'score':>12s} | {'R':>8s}")
        print("-" * 52)

        for c in conds:
            row = wsc.run_condition(model, idx_corr, c, token_a_id, token_b_id)
            score = row["score"]
            R = wsc.normalized_restoration(score, score_clean, score_corr)

            if c.source is None:
                src = "-"
            else:
                src = f"({c.source[0]},{c.source[1]})"

            print(f"{c.name:12s} | {src:10s} | {score:12.6f} | {R:8.4f}")

    # 10) OPTIONAL: build one full wrong-source heatmap using deterministic rule (pos+1 else pos-1)
    print("\nOptional: computing a full wrong-source heatmap with rule: source=(L,P+1) else (L,P-1)")
    ws_heatmap = torch.empty_like(match_heatmap)
    for L in range(n_layers):
        for P in range(seq_len):
            srcP = P + 1 if (P + 1 < seq_len) else (P - 1)
            _ = model(idx_corr, layer_to_patch=L, position_to_patch=P, source_layer=L, source_position=srcP)
            ws_heatmap[L, P] = wsc.score_from_last_logits(model.last_logits[0], token_a_id, token_b_id)

    torch.save(ws_heatmap.cpu(), "wrong_source_posshift_heatmap.pt")
    print("Saved: wrong_source_posshift_heatmap.pt")
    print("\nDone ✅")


if __name__ == "__main__":
    main()
