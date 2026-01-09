import os
import pathlib
import pytest
import torch

import mingpt
import mingpt.model
from mingpt.model import GPT

import repo_orientation as ro


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


@pytest.mark.slow
def test_slow_from_pretrained_gpt2_loads_and_runs():
    # Slow test: tries to download and load GPT-2 weights.
    # If network/cache issues happen in Colab, we skip rather than fail hard.
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
