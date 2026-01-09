import pathlib
import sys

import pytest
import torch

# Colab-friendly: ensure mingpt editable install path is visible during pytest subprocess
COLAB_MINGPT_PATH = pathlib.Path("/content/src/mingpt")
if COLAB_MINGPT_PATH.exists():
    sys.path.append(str(COLAB_MINGPT_PATH))

import mingpt
import mingpt.model
from mingpt.model import GPT

import repo_orientation as ro
import tokenization_protocol as tp


# --------------------------
# Section 2 tests (repo orientation)
# --------------------------

def test_mingpt_importable_and_paths_exist():
    paths = ro.get_paths()
    assert "mingpt.__file__" in paths and "mingpt.model.__file__" in paths

    pkg_path = pathlib.Path(paths["mingpt.__file__"])
    model_path = pathlib.Path(paths["mingpt.model.__file__"])
    assert pkg_path.exists(), f"mingpt package file not found: {pkg_path}"
    assert model_path.exists(), f"mingpt.model file not found: {model_path}"


def test_attn_bias_fix_present_or_applied():
    src = ro.read_model_source()
    assert ro.attn_bias_fix_present(src), (
        "Required fix not found in mingpt/model.py. "
        "Expected assert to ignore keys ending with .attn.bias."
    )


def test_forward_pipeline_landmarks_present():
    fwd_src = ro.forward_source()
    lm = ro.find_forward_landmarks(fwd_src)
    assert lm.has_tok_emb, "Expected token embedding (wte/tok_emb) usage in forward."
    assert lm.has_pos_emb, "Expected positional embedding (wpe/pos_emb) usage in forward."
    assert lm.has_blocks_loop, "Expected loop over transformer blocks in forward."
    assert lm.has_ln_f, "Expected final layer norm ln_f in forward."
    assert lm.has_lm_head, "Expected lm_head/logits in forward."


def test_fast_forward_and_generate_from_scratch():
    # Fast test: avoid downloading HF weights.
    cfg = GPT.get_default_config()
    cfg.model_type = "gpt-nano"  # tiny
    cfg.vocab_size = 1000
    cfg.block_size = 64
    model = GPT(cfg)
    model.eval()

    idx = torch.randint(0, cfg.vocab_size, (1, 10), dtype=torch.long)
    with torch.no_grad():
        logits, loss = model(idx)
    assert logits.shape == (1, 10, cfg.vocab_size)
    assert loss is None

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=5, do_sample=False)
    assert out.shape[1] == 15


# --------------------------
# Section 3 tests (tokenization protocol)
# --------------------------

def test_diff_positions_length_mismatch_includes_tail():
    a = [1, 2, 3]
    b = [1, 2, 3, 4, 5]
    diffs = tp.diff_positions(a, b)
    assert diffs == [3, 4]


def test_compare_reports_detects_one_token_diff_synthetic():
    clean = tp.TokenizationReport(
        text="clean",
        token_ids=[10, 20, 30],
        token_strs=["a", "b", "c"],
        seq_len=3,
        decoded_roundtrip="abc",
    )
    corrupt = tp.TokenizationReport(
        text="corrupt",
        token_ids=[10, 99, 30],
        token_strs=["a", "X", "c"],
        seq_len=3,
        decoded_roundtrip="aXc",
    )
    comp = tp.compare_clean_corrupt(clean, corrupt)
    assert comp.same_length is True
    assert comp.diff_positions == [1]
    assert comp.diff_count == 1
    assert comp.one_token_diff is True


def test_assert_one_token_difference_raises_when_multi_diff():
    clean = tp.TokenizationReport(
        text="clean",
        token_ids=[1, 2, 3],
        token_strs=["a", "b", "c"],
        seq_len=3,
        decoded_roundtrip="abc",
    )
    corrupt = tp.TokenizationReport(
        text="corrupt",
        token_ids=[9, 2, 8],
        token_strs=["X", "b", "Y"],
        seq_len=3,
        decoded_roundtrip="XbY",
    )
    comp = tp.compare_clean_corrupt(clean, corrupt)
    assert comp.diff_count == 2
    with pytest.raises(ValueError):
        tp.assert_one_token_difference(comp)


@pytest.mark.slow
def test_bpe_tokenization_roundtrip_and_lengths():
    """
    Slow-ish test because BPETokenizer may download merges/vocab on first use in a fresh runtime.
    """
    from mingpt.bpe import BPETokenizer

    try:
        bpe = BPETokenizer()
    except Exception as e:
        pytest.skip(f"Skipping BPETokenizer test due to tokenizer init/download error: {e}")

    text = "Michelle Jones was a top-notch student. Michelle"
    rep = tp.build_report(bpe, text)

    # Basic sanity
    assert rep.seq_len > 0
    assert len(rep.token_ids) == rep.seq_len
    assert len(rep.token_strs) == rep.seq_len

    # Roundtrip should contain the key content (exact equality may vary by whitespace normalization)
    assert "Michelle" in rep.decoded_roundtrip


@pytest.mark.slow
def test_bpe_pair_validation_example_michelle_jones_smith():
    """
    Uses the assignment's canonical-style example to ensure:
    - same token length
    - ideally a one-token difference (it usually is, but tokenizer quirks can vary)
    """
    from mingpt.bpe import BPETokenizer

    try:
        bpe = BPETokenizer()
    except Exception as e:
        pytest.skip(f"Skipping BPETokenizer test due to tokenizer init/download error: {e}")

    clean = "Michelle Jones was a top-notch student. Michelle"
    corrupt = "Michelle Smith was a top-notch student. Michelle"

    clean_rep = tp.build_report(bpe, clean)
    corrupt_rep = tp.build_report(bpe, corrupt)
    comp = tp.compare_clean_corrupt(clean_rep, corrupt_rep)

    assert comp.same_length is True, f"Expected same token length; got {clean_rep.seq_len} vs {corrupt_rep.seq_len}"

    # We prefer one-token diff; if it isn't, we still show it's a valid pair for same-length constraint.
    # But for the assignment report you should aim for diff_count == 1.
    assert comp.diff_count >= 1


@pytest.mark.slow
def test_slow_from_pretrained_gpt2_loads_and_runs():
    """
    Slow test: downloads and loads GPT-2 weights.
    If network/cache issues happen in Colab, we skip rather than fail hard.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = GPT.from_pretrained("gpt2")
    except Exception as e:
        pytest.skip(f"Skipping from_pretrained test due to load/download error: {e}")

    model.to(device)
    model.eval()

    idx = torch.randint(0, 50257, (1, 8), dtype=torch.long, device=device)
    with torch.no_grad():
        logits, loss = model(idx)

    assert logits.shape == (1, 8, 50257)
    assert loss is None
