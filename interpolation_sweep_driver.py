from __future__ import annotations

import argparse
from typing import List, Tuple, Optional

import torch

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer

import tokenization_protocol as tp  
from baseline_utils import single_token_id  


def parse_alphas(xs: List[str]) -> List[float]:
    return [float(x) for x in xs]


def score_from_last_logits(last_logits_1d: torch.Tensor, token_a_id: int, token_b_id: int) -> float:
    # score = logit(B) - logit(A)
    return float(last_logits_1d[token_b_id] - last_logits_1d[token_a_id])


@torch.no_grad()
def run_once(
    model: GPT,
    idx: torch.Tensor,
    *,
    cache_clean: bool = False,
    overwrite_cache: bool = False,
    record_acts: bool = False,
    layer_to_patch: Optional[int] = None,
    position_to_patch: Optional[int] = None,
    patch_alpha: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[List[List[torch.Tensor]]]]:
    logits, _ = model(
        idx,
        record_activations=record_acts,
        cache_activations=cache_clean,
        overwrite_cache=overwrite_cache,
        layer_to_patch=layer_to_patch,
        position_to_patch=position_to_patch,
        patch_alpha=patch_alpha,
    )
    if model.last_logits is None:
        raise RuntimeError("model.last_logits was not set.")
    acts = model.last_activations if record_acts else None
    return model.last_logits[0].detach().clone(), acts


def pick_best_layer_at_changed_pos(
    model: GPT,
    idx_corrupt: torch.Tensor,
    token_a_id: int,
    token_b_id: int,
    *,
    changed_pos: int,
    score_corr: float,
) -> int:
    n_layer = len(model.transformer.h)
    best_L = 0
    best_restoration = -1e9

    for L in range(n_layer):
        last, _ = run_once(
            model,
            idx_corrupt,
            layer_to_patch=L,
            position_to_patch=changed_pos,
            patch_alpha=1.0,
        )
        s = score_from_last_logits(last, token_a_id, token_b_id)
        restoration = abs(s - score_corr)
        if restoration > best_restoration:
            best_restoration = restoration
            best_L = L

    return best_L


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", type=str, default="Juan Antonio watched my neural network learn to juggle bananas; he called it wizard science and demanded espresso")
    ap.add_argument("--corrupt", type=str, default="Juan Antonio watched my neural network learn to juggle bananas; he called it algorithm science and demanded espresso")
    ap.add_argument("--token_a", type=str, default=" wizard")
    ap.add_argument("--token_b", type=str, default=" algorithm")

    ap.add_argument("--layer", type=int, default=-1, help="Target layer L. If -1, auto-pick best L at changed token position.")
    ap.add_argument("--pos", type=int, default=-1, help="Target position P. If -1, use changed token position.")
    ap.add_argument("--alphas", nargs="+", default=["0", "0.25", "0.5", "0.75", "1"])

    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--check_mixture", action="store_true", help="Also verify x_patched ≈ αx_clean+(1-α)x_corr using recorded activations.")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    bpe = BPETokenizer()

    # Tokenization validation
    clean_rep = tp.build_report(bpe, args.clean)
    corrupt_rep = tp.build_report(bpe, args.corrupt)
    comp = tp.compare_clean_corrupt(clean_rep, corrupt_rep)

    if not comp.same_length:
        raise RuntimeError(f"Token length mismatch: clean={clean_rep.seq_len}, corrupt={corrupt_rep.seq_len}")
    if comp.diff_count != 1:
        raise RuntimeError(f"Expected exactly 1 differing token position; got {comp.diff_count}: {comp.diff_positions}")

    changed_pos = int(comp.diff_positions[0])
    print(f"Seq len T={clean_rep.seq_len}, changed token position={changed_pos}")

    # Load model
    model = GPT.from_pretrained("gpt2").to(device).eval()

    # Token ids for metric
    token_a_id = single_token_id(bpe, args.token_a)
    token_b_id = single_token_id(bpe, args.token_b)

    # Build tensors
    idx_clean = bpe(args.clean).to(device)
    idx_corrupt = bpe(args.corrupt).to(device)

    # Clean baseline (cache activations)
    last_clean, _ = run_once(model, idx_clean, cache_clean=True, overwrite_cache=True)
    score_clean = score_from_last_logits(last_clean, token_a_id, token_b_id)

    # Corrupt baseline
    last_corr, acts_corr = run_once(model, idx_corrupt, record_acts=args.check_mixture)
    score_corr = score_from_last_logits(last_corr, token_a_id, token_b_id)

    print("\n=== Baselines ===")
    print(f"score_clean  = {score_clean:.6f}")
    print(f"score_corr   = {score_corr:.6f}")
    print(f"delta(corr-clean) = {score_corr - score_clean:.6f}")

    # Choose (L,P)
    P = changed_pos if args.pos < 0 else int(args.pos)
    if args.layer < 0:
        L = pick_best_layer_at_changed_pos(model, idx_corrupt, token_a_id, token_b_id, changed_pos=P, score_corr=score_corr)
        print(f"\nAuto-picked layer L={L} at position P={P}")
    else:
        L = int(args.layer)
        print(f"\nUsing provided (L,P)=({L},{P})")

    alphas = parse_alphas(args.alphas)

    # Full patch score (alpha=1) used for endpoint check
    last_full, _ = run_once(model, idx_corrupt, layer_to_patch=L, position_to_patch=P, patch_alpha=1.0)
    score_full = score_from_last_logits(last_full, token_a_id, token_b_id)

    # Sweep
    print("\n=== Interpolation sweep ===")
    print("alpha | score(alpha) | R_hat_vs_full | meta(last_patch,last_alpha)")
    print("-"*78)

    eps_score = 1e-5
    base_denom = (score_full - score_corr)

    clean_vec = model.clean_activations[L][P].to(device)

    for a in alphas:
        record = bool(args.check_mixture)
        last_a, acts_a = run_once(
            model,
            idx_corrupt,
            record_acts=record,
            layer_to_patch=L,
            position_to_patch=P,
            patch_alpha=a,
        )
        s = score_from_last_logits(last_a, token_a_id, token_b_id)

        # normalized w.r.t full patch (so endpoints are guaranteed to be 0 and 1 if correct)
        if abs(base_denom) < 1e-12:
            rhat = float("nan")
        else:
            rhat = (s - score_corr) / base_denom

        meta = (model.last_patch, model.last_patch_alpha)
        print(f"{a:>4.2f} | {s:>11.6f} | {rhat:>12.6f} | {meta}")

        # Mixture check (activation-level)
        if args.check_mixture:
            if acts_corr is None or acts_a is None:
                raise RuntimeError("Mixture check requested but activations were not recorded.")
            x_corr = acts_corr[L][P].to(device)
            x_pat = acts_a[L][P].to(device)
            target = (a * clean_vec) + ((1.0 - a) * x_corr)
            max_err = float((x_pat - target).abs().max().item())
            if max_err > 1e-4:
                raise RuntimeError(f"Mixture check FAILED at alpha={a}: max|x_patched - mix| = {max_err:.6e}")

    # Endpoint checks
    # alpha=0
    last_0, _ = run_once(model, idx_corrupt, layer_to_patch=L, position_to_patch=P, patch_alpha=0.0)
    score_0 = score_from_last_logits(last_0, token_a_id, token_b_id)

    # alpha=1
    last_1, _ = run_once(model, idx_corrupt, layer_to_patch=L, position_to_patch=P, patch_alpha=1.0)
    score_1 = score_from_last_logits(last_1, token_a_id, token_b_id)

    print("\n=== Endpoint checks ===")
    print(f"score(alpha=0) = {score_0:.6f}  vs score_corr = {score_corr:.6f}")
    print(f"score(alpha=1) = {score_1:.6f}  vs score_full = {score_full:.6f}")

    if abs(score_0 - score_corr) > eps_score:
        raise RuntimeError("FAILED endpoint check α=0: score(0) != score_corr (patch should be a no-op).")
    if abs(score_1 - score_full) > eps_score:
        raise RuntimeError("FAILED endpoint check α=1: score(1) != score_full (must match standard patch).")

    print("\nThis looks correct: endpoints match and sweep ran successfully.")
    if args.check_mixture:
        print("Mixture check passed: activations match αx_clean+(1-α)x_corr at the patched coordinate.")


if __name__ == "__main__":
    main()
