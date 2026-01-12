from __future__ import annotations

import torch

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed

from patching_sweep import build_patching_sweep


# Canonical example (you can change these later for your own creative experiment)
CLEAN_TEXT = "Michelle Jones was a top-notch student. Michelle"
CORRUPT_TEXT = "Michelle Smith was a top-notch student. Michelle"

# IMPORTANT: GPT-2 BPE usually needs the leading space for mid-sequence words
TOKEN_A = " Jones"   # clean-consistent
TOKEN_B = " Smith"   # corrupt-consistent

OUT_MATRIX_PATH = "section9_diff_matrix.pt"        # tensor only (backward compatible)
OUT_RESULT_PATH = "section9_sweep_result.pt"       # dict with metadata (recommended)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def prompt_token_info(bpe: BPETokenizer, text: str):
    """
    Returns:
      token_ids: List[int]
      token_strs: List[str]  (per-token decoded strings)
    """
    ids = bpe(text)[0].tolist()
    token_ids = [int(x) for x in ids]
    token_strs = [bpe.decode(torch.tensor([tid], dtype=torch.long)) for tid in token_ids]
    return token_ids, token_strs


@torch.no_grad()
def main() -> None:
    set_seed(3407)

    device = get_device()
    print("Device:", device)

    model = GPT.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    bpe = BPETokenizer()

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
    print(f"shape: {tuple(res.matrix.shape)} (n_layers, seq_len)")
    print(f"n_layers: {res.n_layers} seq_len: {res.seq_len}")
    print(f"dtype: {res.matrix.dtype} device: {res.matrix.device}")

    # 1) Save tensor only (as you already do)
    torch.save(res.matrix, OUT_MATRIX_PATH)
    print(f"\nSaved: {OUT_MATRIX_PATH}")

    # 2) Save full metadata for later heatmap + report reproducibility
    token_ids, token_strs = prompt_token_info(bpe, CLEAN_TEXT)

    payload = {
        "matrix": res.matrix,  # (n_layers, seq_len), CPU float32
        "axis_convention": {"rows": "layers", "cols": "positions"},
        "n_layers": res.n_layers,
        "seq_len": res.seq_len,
        "token_a_str": res.token_a_str,
        "token_b_str": res.token_b_str,
        "token_a_id": res.token_a_id,
        "token_b_id": res.token_b_id,
        "clean_score": res.clean_score,
        "corrupt_score": res.corrupt_score,
        "delta_corrupt_minus_clean": float(res.corrupt_score - res.clean_score),
        "clean_text": res.clean_text,
        "corrupt_text": res.corrupt_text,
        "prompt_token_ids": token_ids,
        "prompt_token_strs": token_strs,
    }

    torch.save(payload, OUT_RESULT_PATH)
    print(f"Saved (with metadata): {OUT_RESULT_PATH}")


if __name__ == "__main__":
    main()
