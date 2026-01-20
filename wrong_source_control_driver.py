from __future__ import annotations

import math
import torch
from torch.nn import functional as F

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
import tokenization_protocol as tp


def single_token_id(bpe: BPETokenizer, token_str: str) -> int:
    ids = bpe(token_str)[0].tolist()
    if len(ids) != 1:
        raise ValueError(f"{repr(token_str)} is not a single BPE token. Got {len(ids)} ids: {ids}")
    return int(ids[0])


def score_from_last_logits(last_logits_1d: torch.Tensor, id_a: int, id_b: int) -> float:
    # score = logit(B) - logit(A)
    return float(last_logits_1d[id_b] - last_logits_1d[id_a])


def restoration_fraction(score_p: float, score_corr: float, score_clean: float) -> float:
    den = (score_clean - score_corr)
    if abs(den) < 1e-12:
        return float("nan")
    return (score_p - score_corr) / den


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # You can replace these with YOUR final creative pair once ready
    CLEAN_TEXT = "Michelle Jones was a top-notch student. Michelle"
    CORRUPT_TEXT = "Michelle Smith was a top-notch student. Michelle"
    TOKEN_A = " Jones"   # clean-consistent
    TOKEN_B = " Smith"   # corrupt-consistent

    bpe = BPETokenizer()

    # 1) Validate tokenization constraints (same length, one token diff)
    comp = tp.validate_pair(bpe, CLEAN_TEXT, CORRUPT_TEXT, require_same_length=True, require_one_token_diff=True)
    T = comp.clean.seq_len
    changed_pos = comp.diff_positions[0]
    print(f"Seq len T={T}, changed token position={changed_pos}")

    # 2) Load GPT-2
    model = GPT.from_pretrained("gpt2").to(device).eval()

    id_a = single_token_id(bpe, TOKEN_A)
    id_b = single_token_id(bpe, TOKEN_B)

    idx_clean = bpe(CLEAN_TEXT).to(device)
    idx_corr = bpe(CORRUPT_TEXT).to(device)

    # 3) Clean baseline (cache activations)
    _ = model(idx_clean, cache_activations=True, overwrite_cache=True)
    score_clean = score_from_last_logits(model.last_logits[0], id_a, id_b)

    # 4) Corrupted baseline
    _ = model(idx_corr)
    score_corr = score_from_last_logits(model.last_logits[0], id_a, id_b)

    print("\n=== Baselines ===")
    print(f"score_clean  = {score_clean:.4f}")
    print(f"score_corr   = {score_corr:.4f}")
    print(f"gap (clean-corr) = {score_clean - score_corr:.4f}")

    # 5) Compute MATCH heatmap scores (standard patch)
    n_layer = len(model.transformer.h)
    match_scores = torch.empty((n_layer, T), dtype=torch.float32)

    for L in range(n_layer):
        for P in range(T):
            _ = model(idx_corr, layer_to_patch=L, position_to_patch=P)  # source defaults to (L,P)
            s = score_from_last_logits(model.last_logits[0], id_a, id_b)
            match_scores[L, P] = s

    # “How much closer to clean did we get?”
    # improvement = |corr-clean| - |patched-clean|  (bigger is better)
    base_dist = abs(score_corr - score_clean)
    improvements = torch.empty_like(match_scores)
    for L in range(n_layer):
        for P in range(T):
            improvements[L, P] = base_dist - abs(float(match_scores[L, P]) - score_clean)

    flat = improvements.view(-1)
    topk = torch.topk(flat, k=3)
    hot_coords = []
    for idx in topk.indices.tolist():
        L = idx // T
        P = idx % T
        hot_coords.append((L, P))

    # cold cell: improvement closest to 0
    cold_idx = torch.argmin(torch.abs(flat)).item()
    cold_coord = (cold_idx // T, cold_idx % T)

    coords = hot_coords + [cold_coord]

    print("\nSelected coords:")
    for i, (L, P) in enumerate(coords):
        tag = "HOT" if i < 3 else "COLD"
        print(f"  {tag}: (L={L}, P={P})  match_score={float(match_scores[L,P]):.4f}  improvement={float(improvements[L,P]):.4f}")

    # 6) Run wrong-source variants on these coords
    print("\n=== Wrong-source control table ===")
    header = "coord | variant | source(L,P) | score | R (norm restoration) | last_patch | last_patch_source"
    print(header)
    print("-" * len(header))

    def run_variant(L, P, SL, SP, name):
        _ = model(idx_corr, layer_to_patch=L, position_to_patch=P, source_layer=SL, source_position=SP)
        s = score_from_last_logits(model.last_logits[0], id_a, id_b)
        R = restoration_fraction(s, score_corr, score_clean)
        print(f"({L:02d},{P:02d}) | {name:10s} | ({SL:02d},{SP:02d})     | {s: .4f} | {R: .3f}               | {model.last_patch} | {model.last_patch_source}")
        return R

    for (L, P) in coords:
        # MATCH
        _ = model(idx_corr, layer_to_patch=L, position_to_patch=P)
        s_match = score_from_last_logits(model.last_logits[0], id_a, id_b)
        R_match = restoration_fraction(s_match, score_corr, score_clean)
        print(f"({L:02d},{P:02d}) | {'MATCH':10s} | ({L:02d},{P:02d})     | {s_match: .4f} | {R_match: .3f}               | {model.last_patch} | {model.last_patch_source}")

        R_wrongs = []

        # WS-pos+
        if P + 1 < T:
            R_wrongs.append(run_variant(L, P, L, P + 1, "WS-pos+"))
        # WS-pos-
        if P - 1 >= 0:
            R_wrongs.append(run_variant(L, P, L, P - 1, "WS-pos-"))
        # WS-layer+
        if L + 1 < n_layer:
            R_wrongs.append(run_variant(L, P, L + 1, P, "WS-layer+"))
        # WS-layer-
        if L - 1 >= 0:
            R_wrongs.append(run_variant(L, P, L - 1, P, "WS-layer-"))

        if len(R_wrongs) > 0 and (not math.isnan(R_match)):
            S = R_match - max(R_wrongs)
            print(f"-> Specificity index S = R_match - max(R_wrong) = {S:.3f}\n")
        else:
            print("-> Specificity index S = (not available)\n")


if __name__ == "__main__":
    main()
