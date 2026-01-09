import inspect
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict

import pytest
import torch

# If you happen to be in a Colab-like layout, keep this harmless path add.
COLAB_MINGPT_PATH = pathlib.Path("/content/src/mingpt")
if COLAB_MINGPT_PATH.exists():
    sys.path.append(str(COLAB_MINGPT_PATH))

import mingpt
import mingpt.model
from mingpt.model import GPT


# --------------------------
# Section 2: repo orientation sanity
# --------------------------

@dataclass(frozen=True)
class ForwardLandmarks:
    has_tok_emb: bool
    has_pos_emb: bool
    has_blocks_loop: bool
    has_ln_f: bool
    has_lm_head: bool


def get_paths() -> Dict[str, str]:
    pkg_path = pathlib.Path(mingpt.__file__).resolve()
    model_path = pathlib.Path(mingpt.model.__file__).resolve()
    return {
        "mingpt.__file__": str(pkg_path),
        "mingpt.model.__file__": str(model_path),
    }


def read_model_source() -> str:
    model_path = pathlib.Path(mingpt.model.__file__).resolve()
    return model_path.read_text(encoding="utf-8")


def attn_bias_fix_present(model_source: str) -> bool:
    return 'len([k for k in sd if not k.endswith(".attn.bias")])' in model_source


def forward_source() -> str:
    return inspect.getsource(GPT.forward)


def find_forward_landmarks(src: str) -> ForwardLandmarks:
    has_tok_emb = ("tok_emb" in src) and ("wte" in src)
    has_pos_emb = ("pos_emb" in src) and ("wpe" in src)
    has_blocks_loop = ("for block in self.transformer.h" in src) or ("enumerate(self.transformer.h" in src)
    has_ln_f = "ln_f" in src
    has_lm_head = ("lm_head" in src) and ("logits" in src)
    return ForwardLandmarks(
        has_tok_emb=has_tok_emb,
        has_pos_emb=has_pos_emb,
        has_blocks_loop=has_blocks_loop,
        has_ln_f=has_ln_f,
        has_lm_head=has_lm_head,
    )


def test_mingpt_importable_and_paths_exist():
    paths = get_paths()
    assert "mingpt.__file__" in paths and "mingpt.model.__file__" in paths

    pkg_path = pathlib.Path(paths["mingpt.__file__"])
    model_path = pathlib.Path(paths["mingpt.model.__file__"])
    assert pkg_path.exists(), f"mingpt package file not found: {pkg_path}"
    assert model_path.exists(), f"mingpt.model file not found: {model_path}"


def test_attn_bias_fix_present_or_applied():
    src = read_model_source()
    assert attn_bias_fix_present(src), (
        "Required fix not found in mingpt/model.py. "
        "Expected assert to ignore keys ending with .attn.bias."
    )


def test_forward_pipeline_landmarks_present():
    fwd_src = forward_source()
    lm = find_forward_landmarks(fwd_src)
    assert lm.has_tok_emb, "Expected token embedding (wte/tok_emb) usage in forward."
    assert lm.has_pos_emb, "Expected positional embedding (wpe/pos_emb) usage in forward."
    assert lm.has_blocks_loop, "Expected loop over transformer blocks in forward."
    assert lm.has_ln_f, "Expected final layer norm ln_f in forward."
    assert lm.has_lm_head, "Expected lm_head/logits in forward."


def test_fast_forward_and_generate_from_scratch():
    cfg = GPT.get_default_config()
    cfg.model_type = "gpt-nano"
    cfg.vocab_size = 1000
    cfg.block_size = 64
    model = GPT(cfg).eval()

    idx = torch.randint(0, cfg.vocab_size, (1, 10), dtype=torch.long)
    with torch.no_grad():
        logits, loss = model(idx)
    assert logits.shape == (1, 10, cfg.vocab_size)
    assert loss is None

    with torch.no_grad():
        out = model.generate(idx, max_new_tokens=5, do_sample=False)
    assert out.shape[1] == 15


# --------------------------
# Shared helper for Section 5/6 tests
# --------------------------

def _make_tiny_gpt():
    cfg = GPT.get_default_config()
    cfg.model_type = "gpt-nano"
    cfg.vocab_size = 1000
    cfg.block_size = 64
    model = GPT(cfg).eval()
    return model, cfg


# --------------------------
# Section 5: activation recording / clean cache
# --------------------------

def test_section5_cache_structure_and_shapes():
    model, cfg = _make_tiny_gpt()
    T = 12
    idx = torch.randint(0, cfg.vocab_size, (1, T), dtype=torch.long)

    with torch.no_grad():
        logits, _ = model(idx, cache_activations=True, overwrite_cache=True)

    assert model.clean_activations is not None
    assert isinstance(model.clean_activations, list)
    assert len(model.clean_activations) == len(model.transformer.h)  # n_layer
    assert len(model.clean_activations[0]) == T

    d_model = model.transformer.wte.weight.shape[1]
    a00 = model.clean_activations[0][0]
    assert tuple(a00.shape) == (d_model,)
    assert a00.requires_grad is False


def test_section5_cache_uses_detach_clone_not_views():
    model, cfg = _make_tiny_gpt()
    T = 6
    idx = torch.randint(0, cfg.vocab_size, (1, T), dtype=torch.long)

    with torch.no_grad():
        _ = model(idx, cache_activations=True, overwrite_cache=True)

    a0 = model.clean_activations[0][0]
    a1 = model.clean_activations[0][1]
    assert a0.data_ptr() != a1.data_ptr(), "Expected clone()d per-position tensors with distinct storage."


def test_section5_logits_identical_with_and_without_recording():
    model, cfg = _make_tiny_gpt()
    T = 10
    idx = torch.randint(0, cfg.vocab_size, (1, T), dtype=torch.long)

    with torch.no_grad():
        logits1, _ = model(idx)  # normal
        logits2, _ = model(idx, record_activations=True, cache_activations=False)  # recording only

    assert torch.allclose(logits1, logits2), "Activation recording must not change logits."


def test_section5_clean_cache_not_overwritten_unless_requested():
    model, cfg = _make_tiny_gpt()
    idx1 = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)
    idx2 = torch.randint(0, cfg.vocab_size, (1, 8), dtype=torch.long)

    with torch.no_grad():
        _ = model(idx1, cache_activations=True, overwrite_cache=True)

    snap = [[t.clone() for t in layer] for layer in model.clean_activations]

    with torch.no_grad():
        _ = model(idx2)
    for L in range(len(snap)):
        for p in range(len(snap[L])):
            assert torch.equal(model.clean_activations[L][p], snap[L][p])

    with torch.no_grad():
        _ = model(idx2, record_activations=True, cache_activations=False)
    for L in range(len(snap)):
        for p in range(len(snap[L])):
            assert torch.equal(model.clean_activations[L][p], snap[L][p])

    with pytest.raises(RuntimeError):
        with torch.no_grad():
            _ = model(idx2, cache_activations=True, overwrite_cache=False)


def test_section5_batch_behavior_records_only_first_element():
    model1, cfg = _make_tiny_gpt()
    model2, _ = _make_tiny_gpt()
    model2.load_state_dict(model1.state_dict())  # identical weights

    T = 9
    idx_batch = torch.randint(0, cfg.vocab_size, (2, T), dtype=torch.long)
    idx_first = idx_batch[:1, :]

    with torch.no_grad():
        _ = model1(idx_batch, cache_activations=True, overwrite_cache=True)
        _ = model2(idx_first, cache_activations=True, overwrite_cache=True)

    for L in range(len(model1.clean_activations)):
        for p in range(T):
            assert torch.allclose(
                model1.clean_activations[L][p],
                model2.clean_activations[L][p],
                rtol=1e-5,
                atol=1e-6,
            )


# --------------------------
# Section 6: last-token logits extraction (NEW)
# --------------------------

def test_section6_last_logits_exists_and_matches_last_position_logits():
    model, cfg = _make_tiny_gpt()
    T = 11
    idx = torch.randint(0, cfg.vocab_size, (1, T), dtype=torch.long)

    with torch.no_grad():
        logits, _ = model(idx)

    assert hasattr(model, "last_logits"), "Expected GPT to expose model.last_logits"
    assert model.last_logits is not None, "model.last_logits was not set by forward()"
    assert tuple(model.last_logits.shape) == (1, cfg.vocab_size)

    expected = logits[:, -1, :]
    assert torch.allclose(model.last_logits, expected), "model.last_logits must equal logits[:, -1, :] for the same run"


def test_section6_last_logits_is_detached_and_cloned():
    model, cfg = _make_tiny_gpt()
    T = 7
    idx = torch.randint(0, cfg.vocab_size, (1, T), dtype=torch.long)

    with torch.no_grad():
        logits, _ = model(idx)

    view = logits[:, -1, :]  # view into logits storage
    assert model.last_logits.requires_grad is False
    # clone must not share underlying storage with the view
    assert model.last_logits.data_ptr() != view.data_ptr(), "Expected last_logits to be a clone(), not a view"


def test_section6_last_logits_computed_even_when_recording_or_caching():
    model, cfg = _make_tiny_gpt()
    T = 9
    idx = torch.randint(0, cfg.vocab_size, (1, T), dtype=torch.long)

    with torch.no_grad():
        logits_a, _ = model(idx, record_activations=True, cache_activations=False)
    assert model.last_logits is not None
    assert torch.allclose(model.last_logits, logits_a[:, -1, :])

    with torch.no_grad():
        logits_b, _ = model(idx, cache_activations=True, overwrite_cache=True)
    assert model.last_logits is not None
    assert torch.allclose(model.last_logits, logits_b[:, -1, :])
