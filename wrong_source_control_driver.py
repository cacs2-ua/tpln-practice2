from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from mingpt.bpe import BPETokenizer
from mingpt.model import GPT
from mingpt.utils import set_seed


@dataclass(frozen=True)
class PatchResult:
    target: Tuple[int, int]
    variant: str
    source: Tuple[int, int]
    score: float
    R: float           # normalized restoration (can be > 1 if overshoot)
    C: float           # normalized closeness-to-clean (1 is best)
    last_patch: Optional[Tuple[int, int]]
    last_patch_source: Optional[Tuple[int, int]]


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def single_token_id(bpe: BPETokenizer, token_str: str) -> int:
    ids = bpe(token_str)[0].tolist()
    if len(ids) != 1:
        raise ValueError(
            f"Token string must map to exactly 1 BPE token. "
            f"Got {len(ids)} tokens for {repr(token_str)}: {ids}"
        )
    return int(ids[0])


def logit_diff_from_last_logits(last_logits_1d: torch.Tensor, token_b_id: int, token_a_id: int) -> float:
    # score = logit(B) - logit(A)
    return float(last_logits_1d[token_b_id] - last_logits_1d[token_a_id])


def norm_restoration(score: float, score_clean: float, score_corr: float) -> float:
    # R = (score - score_corr) / (score_clean - score_corr)
    denom = (score_clean - score_corr)
    if abs(denom) < 1e-12:
        return 0.0
    return (score - score_corr) / denom


def norm_closeness(score: float, score_clean: float, score_corr: float) -> float:
    # C = 1 - |score - score_clean| / |score_corr - score_clean|
    denom = abs(score_corr - score_clean)
    if denom < 1e-12:
        return 1.0
    return 1.0 - (abs(score - score_clean) / denom)


@torch.no_grad()
def run_score(
    model: GPT,
    idx: torch.Tensor,
    token_a_id: int,
    token_b_id: int,
    *,
    layer_to_patch: Optional[int] = None,
    position_to_patch: Optional[int] = None,
    source_layer: Optional[int] = None,
    source_position: Optional[int] = None,
) -> float:
    _logits, _loss = model(
        idx,
        record_activations=False,
        cache_activations=False,
        overwrite_cache=False,
        layer_to_patch=layer_to_patch,
        position_to_patch=position_to_patch,
        source_layer=source_layer,
        source_position=source_position,
    )
    if model.last_logits is None:
        raise RuntimeError("model.last_logits is None after forward().")
    return logit_diff_from_last_logits(model.last_logits[0], token_b_id=token_b_id, token_a_id=token_a_id)


def build_wrong_source_variants(L: int, P: int, n_layer: int, T: int) -> List[Tuple[str, int, int]]:
    variants: List[Tuple[str, int, int]] = []
    variants.append(("MATCH", L, P))

    # Position mismatch (same layer)
    if P + 1 < T:
        variants.append(("WS-pos+", L, P + 1))
    if P - 1 >= 0:
        variants.append(("WS-pos-", L, P - 1))

    # Layer mismatch (same position)
    if L + 1 < n_layer:
        variants.append(("WS-layer+", L + 1, P))
    if L - 1 >= 0:
        variants.append(("WS-layer-", L - 1, P))

    return variants


def select_hotspots(match_scores: torch.Tensor, score_clean: float, k: int = 3) -> List[Tuple[int, int]]:
    # Pick coords whose MATCH patched score is closest to clean (min |score - score_clean|)
    n_layer, T = match_scores.shape
    flat: List[Tuple[float, int, int]] = []
    for L in range(n_layer):
        for P in range(T):
            d = abs(float(match_scores[L, P]) - score_clean)
            flat.append((d, L, P))
    flat.sort(key=lambda x: x[0])
    out: List[Tuple[int, int]] = []
    for _, L, P in flat:
        out.append((L, P))
        if len(out) >= k:
            break
    return out


def select_coldcell(
    match_scores: torch.Tensor,
    score_corr: float,
    *,
    changed_pos: int,
) -> Tuple[int, int]:
    # Pick coord with minimal |score - score_corr| but avoid positions before/at the changed token
    n_layer, T = match_scores.shape
    pos_min = min(T - 1, changed_pos + 1)

    candidates: List[Tuple[float, int, int]] = []
    for L in range(n_layer):
        for P in range(pos_min, T):
            d = abs(float(match_scores[L, P]) - score_corr)
            candidates.append((d, L, P))

    # Fallback if pos_min kills all candidates
    if not candidates:
        for L in range(n_layer):
            for P in range(T):
                d = abs(float(match_scores[L, P]) - score_corr)
                candidates.append((d, L, P))

    candidates.sort(key=lambda x: x[0])
    _, L, P = candidates[0]
    return (L, P)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean", type=str, default="Michelle Jones was a top-notch student. Michelle")
    p.add_argument("--corrupt", type=str, default="Michelle Smith was a top-notch student. Michelle")
    p.add_argument("--token_a", type=str, default=" Jones", help="Token A (clean-consistent), usually with leading space")
    p.add_argument("--token_b", type=str, default=" Smith", help="Token B (corrupt-consistent), usually with leading space")
    p.add_argument("--seed", type=int, default=3407)
    p.add_argument("--n_hot", type=int, default=3)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device()
    print("Device:", device)

    bpe = BPETokenizer()
    token_a_id = single_token_id(bpe, args.token_a)
    token_b_id = single_token_id(bpe, args.token_b)

    idx_clean = bpe(args.clean).to(device)
    idx_corr = bpe(args.corrupt).to(device)

    if idx_clean.shape != idx_corr.shape:
        raise ValueError(
            f"Clean/corrupt token length mismatch: clean T={idx_clean.shape[1]} vs corrupt T={idx_corr.shape[1]}"
        )

    # Find changed token position (expect exactly one token differs)
    clean_ids = idx_clean[0].tolist()
    corr_ids = idx_corr[0].tolist()
    diffs = [i for i, (a, b) in enumerate(zip(clean_ids, corr_ids)) if int(a) != int(b)]
    if len(diffs) != 1:
        raise ValueError(f"Expected exactly 1 differing token position, found {len(diffs)}: {diffs}")
    changed_pos = int(diffs[0])
    T = int(idx_clean.shape[1])

    model = GPT.from_pretrained("gpt2").to(device).eval()
    n_layer = int(len(model.transformer.h))

    print(f"Seq len T={T}, changed token position={changed_pos}")

    # --- Baselines ---
    _ = model(idx_clean, cache_activations=True, overwrite_cache=True)
    if model.last_logits is None:
        raise RuntimeError("model.last_logits not set on clean run.")
    score_clean = logit_diff_from_last_logits(model.last_logits[0], token_b_id=token_b_id, token_a_id=token_a_id)

    score_corr = run_score(model, idx_corr, token_a_id=token_a_id, token_b_id=token_b_id)

    print("\n=== Baselines ===")
    print(f"score_clean  = {score_clean:.4f}")
    print(f"score_corr   = {score_corr:.4f}")
    print(f"gap (clean-corr) = {(score_clean - score_corr):.4f}")

    # --- Build MATCH heatmap (scores) ---
    match_scores = torch.empty((n_layer, T), dtype=torch.float32)
    for L in range(n_layer):
        for P in range(T):
            s = run_score(
                model,
                idx_corr,
                token_a_id=token_a_id,
                token_b_id=token_b_id,
                layer_to_patch=L,
                position_to_patch=P,
                source_layer=L,
                source_position=P,
            )
            match_scores[L, P] = float(s)

    # --- Select coords: hot + cold ---
    hot = select_hotspots(match_scores, score_clean=score_clean, k=args.n_hot)
    cold = select_coldcell(match_scores, score_corr=score_corr, changed_pos=changed_pos)

    print("\nSelected coords:")
    for (L, P) in hot:
        s = float(match_scores[L, P])
        improvement = (score_corr - s)
        print(f"  HOT: (L={L}, P={P})  match_score={s:.4f}  improvement={improvement:.4f}")
    s_cold = float(match_scores[cold[0], cold[1]])
    print(f"  COLD: (L={cold[0]}, P={cold[1]})  match_score={s_cold:.4f}  improvement={(score_corr - s_cold):.4f}")

    # --- Wrong-source control table ---
    print("\n=== Wrong-source control table ===")
    print("coord | variant | source(L,P) | score | R (restoration) | C (closeness) | last_patch | last_patch_source")
    print("-" * 110)

    selected = hot + [cold]

    for (L, P) in selected:
        variants = build_wrong_source_variants(L, P, n_layer=n_layer, T=T)
        results: List[PatchResult] = []

        for (name, sL, sP) in variants:
            s = run_score(
                model,
                idx_corr,
                token_a_id=token_a_id,
                token_b_id=token_b_id,
                layer_to_patch=L,
                position_to_patch=P,
                source_layer=sL,
                source_position=sP,
            )
            R = norm_restoration(s, score_clean=score_clean, score_corr=score_corr)
            C = norm_closeness(s, score_clean=score_clean, score_corr=score_corr)

            results.append(
                PatchResult(
                    target=(L, P),
                    variant=name,
                    source=(sL, sP),
                    score=float(s),
                    R=float(R),
                    C=float(C),
                    last_patch=model.last_patch,
                    last_patch_source=model.last_patch_source,
                )
            )

        # Print rows
        for r in results:
            print(
                f"({r.target[0]:02d},{r.target[1]:02d}) | "
                f"{r.variant:<9} | "
                f"({r.source[0]:02d},{r.source[1]:02d})     | "
                f"{r.score:>7.4f} | "
                f"{r.R:>7.3f}        | "
                f"{r.C:>7.3f}        | "
                f"{r.last_patch} | {r.last_patch_source}"
            )

        # Specificity index (FIX): use closeness-to-clean, not restoration fraction
        c_match = max([rr.C for rr in results if rr.variant == "MATCH"], default=0.0)
        c_wrong = [rr.C for rr in results if rr.variant != "MATCH"]
        if c_wrong:
            S = c_match - max(c_wrong)
        else:
            S = 0.0

        print(f"-> Specificity index S = C_match - max(C_wrong) = {S:.3f}\n")


if __name__ == "__main__":
    main()
