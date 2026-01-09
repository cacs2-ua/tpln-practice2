import pytest
import torch
from mingpt.model import GPT

def _make_tiny():
    cfg = GPT.get_default_config()
    cfg.model_type = "gpt-nano"
    cfg.vocab_size = 200
    cfg.block_size = 32
    model = GPT(cfg).eval()
    return model, cfg

def _make_clean_corrupt(cfg, T=12):
    clean = torch.randint(0, cfg.vocab_size, (1, T), dtype=torch.long)
    corrupt = clean.clone()
    # change exactly one token id so activations differ
    corrupt[0, 3] = (corrupt[0, 3] + 1) % cfg.vocab_size
    return clean, corrupt

# -----------------------
# Section 5/6 sanity (minimal, to ensure Section 7 didn't break them)
# -----------------------

def test_last_logits_exists_and_shape():
    model, cfg = _make_tiny()
    idx = torch.randint(0, cfg.vocab_size, (1, 10), dtype=torch.long)
    with torch.no_grad():
        logits, _ = model(idx)
    assert model.last_logits is not None
    assert tuple(model.last_logits.shape) == (1, cfg.vocab_size)
    assert torch.allclose(model.last_logits, logits[:, -1, :])

def test_clean_cache_written_only_when_requested():
    model, cfg = _make_tiny()
    clean, corrupt = _make_clean_corrupt(cfg, T=10)

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    snap = [[t.clone() for t in layer] for layer in model.clean_activations]

    # normal run must not mutate clean cache
    with torch.no_grad():
        _ = model(corrupt)
    for L in range(len(snap)):
        for p in range(len(snap[L])):
            assert torch.equal(model.clean_activations[L][p], snap[L][p])

# -----------------------
# Section 7 patching tests
# -----------------------

def test_patch_requires_existing_clean_cache():
    model, cfg = _make_tiny()
    _, corrupt = _make_clean_corrupt(cfg, T=10)

    with pytest.raises(RuntimeError):
        with torch.no_grad():
            _ = model(corrupt, layer_to_patch=0, position_to_patch=3)

def test_patch_argument_pairing_rules():
    model, cfg = _make_tiny()
    clean, corrupt = _make_clean_corrupt(cfg, T=10)

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    with pytest.raises(ValueError):
        with torch.no_grad():
            _ = model(corrupt, layer_to_patch=0, position_to_patch=None)

    with pytest.raises(ValueError):
        with torch.no_grad():
            _ = model(corrupt, layer_to_patch=None, position_to_patch=3)

def test_patch_disallows_cache_write_flags():
    model, cfg = _make_tiny()
    clean, corrupt = _make_clean_corrupt(cfg, T=10)

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    with pytest.raises(RuntimeError):
        with torch.no_grad():
            _ = model(corrupt, layer_to_patch=0, position_to_patch=3, cache_activations=True)

    with pytest.raises(RuntimeError):
        with torch.no_grad():
            _ = model(corrupt, layer_to_patch=0, position_to_patch=3, overwrite_cache=True)

def test_patch_applies_at_exact_layer_and_position_and_only_there():
    model, cfg = _make_tiny()
    clean, corrupt = _make_clean_corrupt(cfg, T=12)

    # Cache clean activations
    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    # Corrupted baseline with recording (to compare)
    with torch.no_grad():
        _ = model(corrupt, record_activations=True, cache_activations=False)
    baseline_acts = model.last_activations
    assert baseline_acts is not None

    # Patched run with recording
    L = 0
    P = 3
    with torch.no_grad():
        _ = model(corrupt, record_activations=True, layer_to_patch=L, position_to_patch=P)
    patched_acts = model.last_activations
    assert patched_acts is not None

    # 1) patched location equals clean cache at that (layer, pos)
    assert torch.allclose(
        patched_acts[L][P],
        model.clean_activations[L][P],
        rtol=1e-5,
        atol=1e-6,
    )

    # 2) baseline at that location differs from clean cache (should, because corrupt differs)
    assert not torch.allclose(
        baseline_acts[L][P],
        model.clean_activations[L][P],
        rtol=1e-5,
        atol=1e-6,
    )

    # 3) same layer, other positions are unchanged by patch at that layer output
    other_pos = 0 if P != 0 else 1
    assert torch.allclose(
        patched_acts[L][other_pos],
        baseline_acts[L][other_pos],
        rtol=1e-5,
        atol=1e-6,
    )

    # 4) bookkeeping says exactly one patch
    assert model.last_patch == (L, P)

def test_patch_changes_last_logits_vs_corrupted_baseline():
    model, cfg = _make_tiny()
    clean, corrupt = _make_clean_corrupt(cfg, T=12)

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    with torch.no_grad():
        _ = model(corrupt)
    base_last = model.last_logits.clone()

    with torch.no_grad():
        _ = model(corrupt, layer_to_patch=0, position_to_patch=3)
    patched_last = model.last_logits.clone()

    # Almost surely different if patch actually applied
    assert not torch.allclose(base_last, patched_last)

def test_clean_cache_not_mutated_by_patched_runs():
    model, cfg = _make_tiny()
    clean, corrupt = _make_clean_corrupt(cfg, T=12)

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    snap = [[t.clone() for t in layer] for layer in model.clean_activations]

    with torch.no_grad():
        _ = model(corrupt, layer_to_patch=0, position_to_patch=3, record_activations=True)

    for L in range(len(snap)):
        for p in range(len(snap[L])):
            assert torch.equal(model.clean_activations[L][p], snap[L][p])
