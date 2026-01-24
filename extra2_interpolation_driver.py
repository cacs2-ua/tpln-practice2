from __future__ import annotations

import argparse
from typing import List, Tuple

import torch
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer

import interpolation_sweep as isweep


def parse_coords(s: str) -> List[Tuple[int, int]]:
    # "L:P,L:P" -> [(L,P),...]
    out = []
    s = s.strip()
    if not s:
        return out
    for part in s.split(","):
        Ls, Ps = part.strip().split(":")
        out.append((int(Ls), int(Ps)))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clean", type=str, default="Juan Antonio watched my neural network learn to juggle bananas; he called it wizard science and demanded espresso")
    p.add_argument("--corrupt", type=str, default="Juan Antonio watched my neural network learn to juggle bananas; he called it algorithm science and demanded espresso")
    p.add_argument("--token_a", type=str, default=" wizard")   # clean-consistent
    p.add_argument("--token_b", type=str, default=" algorithm")   # corrupt-consistent
    p.add_argument("--alphas", type=str, default="0,0.25,0.5,0.75,1")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--coords", type=str, default="", help="Optional manual coords 'L:P,L:P,...' (skips hotspot search)")
    p.add_argument("--out", type=str, default="extra2_interpolation_curves.png")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = GPT.from_pretrained("gpt2").to(device).eval()
    bpe = BPETokenizer()

    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    # 1) baselines + cache clean activations
    base = isweep.compute_baselines(
        model,
        bpe,
        clean_text=args.clean,
        corrupt_text=args.corrupt,
        token_a_str=args.token_a,
        token_b_str=args.token_b,
        device=device,
        overwrite_cache=True,
    )
    print("\n=== Baselines ===")
    print(f"score_clean  = {base.clean_score:.4f}")
    print(f"score_corr   = {base.corrupt_score:.4f}")
    print(f"Seq len T    = {base.seq_len}")
    print(f"n_layer      = {base.n_layer}")

    manual_coords = parse_coords(args.coords)
    if manual_coords:
        hotspots = manual_coords
        coldspot = manual_coords[-1]
        print("\nUsing manual coords:", hotspots)
    else:
        # 2) compute alpha=1 patch matrix (standard heatmap values)
        print("\nComputing full alpha=1 patch matrix (for hotspot selection)...")
        M = isweep.full_patch_matrix(model, bpe, args.corrupt, base, device=device)

        # 3) pick hotspots + coldspot
        hotspots, coldspot = isweep.select_hotspots(M, base, top_k=args.top_k)
        print("Hotspots:", hotspots)
        print("Coldspot:", coldspot)

    # 4) interpolation sweeps
    curves = []
    for c in hotspots:
        curves.append(isweep.interpolation_curve(model, bpe, args.corrupt, base, c, alphas, device=device))
    # add control curve
    if coldspot not in hotspots:
        curves.append(isweep.interpolation_curve(model, bpe, args.corrupt, base, coldspot, alphas, device=device))

    print("\n=== Curves (R(alpha)) ===")
    for cv in curves:
        L, P = cv.coord
        print(f"\nCoord (L={L}, P={P}) alpha50={cv.alpha50}")
        for a, s, r in zip(cv.alphas, cv.scores, cv.restorations):
            print(f"  alpha={a:>4.2f}  score={s:>8.4f}  R={r:>8.4f}")

    # 5) plot
    out_path = isweep.plot_restoration_curves(curves, out_path=args.out)
    print(f"\nSaved plot to: {out_path}")


if __name__ == "__main__":
    main()
