import pytest
import torch
from torch.nn import functional as F

from mingpt.model import GPT

import baseline_utils as bu


# -------------------------
# Dummy tokenizer (FAST tests, no downloads)
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
        # expects shape (1,) or (n,)
        ids = t.flatten().tolist()
        return "".join(self.inv.get(int(i), f"<{int(i)}>") for i in ids)


def _make_tiny_gpt(vocab_size=50, block_size=32):
    cfg = GPT.get_default_config()
    cfg.model_type = "gpt-nano"
    cfg.vocab_size = vocab_size
    cfg.block_size = block_size
    return GPT(cfg).eval()


# -------------------------
# Section 8: baseline utils tests (FAST)
# -------------------------

def test_single_token_id_accepts_single_and_rejects_multi():
    bpe = DummyTokenizer()
    assert bu.single_token_id(bpe, " A") == 7
    assert bu.single_token_id(bpe, " B") == 8

    # Multi-token string under DummyTokenizer (tokenizes as chars)
    with pytest.raises(ValueError):
        _ = bu.single_token_id(bpe, "ab")


def test_compute_logit_diff_correctness():
    last = torch.tensor([0.0, 1.0, 2.0, -3.0], dtype=torch.float32)
    logit_a, logit_b, score = bu.compute_logit_diff(last, token_b_id=2, token_a_id=1)
    assert logit_a == 1.0
    assert logit_b == 2.0
    assert score == 1.0


def test_topk_from_last_logits_sorted_and_probabilities_valid():
    bpe = DummyTokenizer()
    last = torch.tensor([0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 0.5, 0.25, 0.1], dtype=torch.float32)
    k = 5
    topk = bu.topk_from_last_logits(bpe, last, k=k)

    assert len(topk) == k
    # probabilities in [0,1] and sorted desc
    probs = [e.prob for e in topk]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert probs == sorted(probs, reverse=True)


def test_clean_baseline_caches_activations_and_sets_last_logits():
    model = _make_tiny_gpt(vocab_size=60, block_size=16)
    bpe = DummyTokenizer()

    clean_text = "abc"      # -> ids [1,2,3]
    token_a = " A"          # -> id 7
    token_b = " B"          # -> id 8

    res = bu.run_clean_baseline(model, bpe, clean_text, token_a, token_b, device="cpu", top_k=5, overwrite_cache=True)

    assert model.clean_activations is not None, "Clean baseline must cache activations."
    assert model.last_logits is not None, "Clean baseline must set last_logits."
    assert isinstance(res.score_logit_diff, float)


def test_corrupt_baseline_does_not_overwrite_clean_cache():
    model = _make_tiny_gpt(vocab_size=60, block_size=16)
    bpe = DummyTokenizer()

    clean_text = "abc"
    corrupt_text = "abx"  # differs in one char token id

    token_a = " A"
    token_b = " B"

    _ = bu.run_clean_baseline(model, bpe, clean_text, token_a, token_b, device="cpu", top_k=5, overwrite_cache=True)
    snap = [[t.clone() for t in layer] for layer in model.clean_activations]

    _ = bu.run_corrupt_baseline(model, bpe, corrupt_text, token_a, token_b, device="cpu", top_k=5)

    # verify cache unchanged
    for L in range(len(snap)):
        for p in range(len(snap[L])):
            assert torch.equal(model.clean_activations[L][p], snap[L][p])


# -------------------------
# Optional SLOW integration test with real GPT-2 (skips if download/cache missing)
# -------------------------

@pytest.mark.slow
def test_section8_metric_shifts_toward_corrupted_on_gpt2_if_available():
    """
    Expected behavior for the canonical example:
      score = logit(' Smith') - logit(' Jones')
      clean prompt  -> score tends to be smaller (often negative)
      corrupt prompt -> score tends to be larger (often positive)
    We only assert corrupt_score > clean_score.
    """
    try:
        from mingpt.bpe import BPETokenizer
        from mingpt.model import GPT
    except Exception as e:
        pytest.skip(f"Skipping due to import error: {e}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = GPT.from_pretrained("gpt2").to(device).eval()
        bpe = BPETokenizer()
    except Exception as e:
        pytest.skip(f"Skipping GPT-2 integration (weights/tokenizer unavailable): {e}")

    CLEAN = "Michelle Jones was a top-notch student. Michelle"
    CORR  = "Michelle Smith was a top-notch student. Michelle"
    A = " Jones"
    B = " Smith"

    clean_res = bu.run_clean_baseline(model, bpe, CLEAN, A, B, device=device, top_k=10, overwrite_cache=True)
    corr_res  = bu.run_corrupt_baseline(model, bpe, CORR, A, B, device=device, top_k=10)

    assert corr_res.score_logit_diff > clean_res.score_logit_diff
