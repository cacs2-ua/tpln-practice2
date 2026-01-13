import pytest
import torch

from mingpt.model import GPT

import baseline_utils as bu
import patching_sweep as ps


# -------------------------
# Dummy tokenizer (FAST, no downloads)
# -------------------------

class DummyTokenizer:
    """
    Minimal stand-in for BPETokenizer that supports:
    - __call__(text) -> tensor(1,T)
    - decode(tensor([id])) -> string
    """
    def __init__(self):
        # ids for prompt chars
        self.map = {
            "a": 1, "b": 2, "c": 3, "x": 4, "y": 5, "z": 6,
            # ids for "tokens" A/B (single-token strings)
            " A": 7,
            " B": 8,
        }
        self.inv = {v: k for k, v in self.map.items()}

    def __call__(self, s: str):
        if s in self.map:
            ids = [self.map[s]]
        else:
            ids = [self.map[ch] for ch in s]  # tokenize per char for prompt
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, t: torch.Tensor) -> str:
        ids = t.flatten().tolist()
        return "".join(self.inv.get(int(i), f"<{int(i)}>") for i in ids)


def _make_tiny_gpt(vocab_size=80, block_size=32):
    cfg = GPT.get_default_config()
    cfg.model_type = "gpt-nano"
    cfg.vocab_size = vocab_size
    cfg.block_size = block_size
    return GPT(cfg).eval()


# -------------------------
# Section 8 baseline utils (fast)
# -------------------------

def test_single_token_id_accepts_single_and_rejects_multi():
    bpe = DummyTokenizer()
    assert bu.single_token_id(bpe, " A") == 7
    assert bu.single_token_id(bpe, " B") == 8

    with pytest.raises(ValueError):
        _ = bu.single_token_id(bpe, "ab")  # multi-token under DummyTokenizer


def test_clean_baseline_caches_activations_and_sets_last_logits():
    model = _make_tiny_gpt(vocab_size=60, block_size=16)
    bpe = DummyTokenizer()

    res = bu.run_clean_baseline(
        model,
        bpe,
        clean_text="abc",
        token_a_str=" A",
        token_b_str=" B",
        device="cpu",
        top_k=5,
        overwrite_cache=True,
    )

    assert model.clean_activations is not None
    assert model.last_logits is not None
    assert isinstance(res.score_logit_diff, float)


def test_corrupt_baseline_does_not_overwrite_clean_cache():
    model = _make_tiny_gpt(vocab_size=60, block_size=16)
    bpe = DummyTokenizer()

    _ = bu.run_clean_baseline(model, bpe, "abc", " A", " B", device="cpu", top_k=5, overwrite_cache=True)
    snap = [[t.clone() for t in layer] for layer in model.clean_activations]

    _ = bu.run_corrupt_baseline(model, bpe, "abx", " A", " B", device="cpu", top_k=5)

    for L in range(len(snap)):
        for p in range(len(snap[L])):
            assert torch.equal(model.clean_activations[L][p], snap[L][p])


# -------------------------
# Section 9 sweep tests (fast, no downloads)
# -------------------------

def test_section9_sweep_matrix_shape_and_dtype():
    model = _make_tiny_gpt(vocab_size=100, block_size=32)

    T = 10
    clean = torch.randint(0, 100, (1, T), dtype=torch.long)
    corrupt = clean.clone()
    corrupt[0, 3] = (corrupt[0, 3] + 1) % 100  # minimal corruption, same length

    # cache clean activations
    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    # sweep (token ids arbitrary for test)
    mat = ps.sweep_from_ids(model, corrupt, token_a_id=1, token_b_id=2, progress=False)

    assert tuple(mat.shape) == (len(model.transformer.h), T)
    assert mat.dtype == torch.float32
    assert mat.device.type == "cpu"
    assert torch.isfinite(mat).all()


def test_section9_sweep_does_not_mutate_clean_cache():
    model = _make_tiny_gpt(vocab_size=100, block_size=32)

    T = 8
    clean = torch.randint(0, 100, (1, T), dtype=torch.long)
    corrupt = clean.clone()
    corrupt[0, 2] = (corrupt[0, 2] + 1) % 100

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    snap = [[t.clone() for t in layer] for layer in model.clean_activations]

    _ = ps.sweep_from_ids(model, corrupt, token_a_id=1, token_b_id=2, progress=False)

    for L in range(len(snap)):
        for p in range(T):
            assert torch.equal(model.clean_activations[L][p], snap[L][p])


def test_section9_sweep_matches_direct_single_call():
    model = _make_tiny_gpt(vocab_size=120, block_size=32)

    T = 9
    clean = torch.randint(0, 120, (1, T), dtype=torch.long)
    corrupt = clean.clone()
    corrupt[0, 4] = (corrupt[0, 4] + 7) % 120

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    token_a_id, token_b_id = 5, 6

    mat = ps.sweep_from_ids(model, corrupt, token_a_id=token_a_id, token_b_id=token_b_id, progress=False)

    # pick one coordinate and verify equality vs direct model call
    L, P = 0, 4
    with torch.no_grad():
        _ = model(corrupt, layer_to_patch=L, position_to_patch=P)
        last = model.last_logits[0].detach()
        direct = ps.logit_diff_from_last_logits(last, token_a_id=token_a_id, token_b_id=token_b_id)

    assert abs(float(mat[L, P]) - float(direct)) < 1e-6


# -------------------------
# Optional SLOW integration with real GPT-2 (skips if unavailable)
# -------------------------

@pytest.mark.slow
def test_section9_real_gpt2_sweep_shape_if_available():
    try:
        from mingpt.bpe import BPETokenizer
    except Exception as e:
        pytest.skip(f"BPETokenizer unavailable: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = GPT.from_pretrained("gpt2").to(device).eval()
        bpe = BPETokenizer()
    except Exception as e:
        pytest.skip(f"GPT-2 weights/tokenizer unavailable: {e}")

    clean = "Michelle Jones was a top-notch student. Michelle"
    corr  = "Michelle Smith was a top-notch student. Michelle"
    A = " Jones"
    B = " Smith"

    res = ps.build_patching_sweep(
        model,
        bpe,
        clean_text=clean,
        corrupt_text=corr,
        token_a_str=A,
        token_b_str=B,
        overwrite_cache=True,
        progress=False,
    )

    assert tuple(res.matrix.shape) == (12, res.seq_len)
    assert torch.isfinite(res.matrix).all()
