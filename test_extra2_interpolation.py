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


def _make_clean_corrupt(cfg, T=12, changed_pos=3):
    torch.manual_seed(0)
    clean = torch.randint(0, cfg.vocab_size, (1, T), dtype=torch.long)
    corrupt = clean.clone()
    corrupt[0, changed_pos] = (corrupt[0, changed_pos] + 1) % cfg.vocab_size
    return clean, corrupt, changed_pos


def test_patch_alpha_bounds_enforced():
    model, cfg = _make_tiny()
    clean, corrupt, P = _make_clean_corrupt(cfg, T=10, changed_pos=3)

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    with pytest.raises(ValueError):
        with torch.no_grad():
            _ = model(corrupt, layer_to_patch=0, position_to_patch=P, patch_alpha=-0.1)

    with pytest.raises(ValueError):
        with torch.no_grad():
            _ = model(corrupt, layer_to_patch=0, position_to_patch=P, patch_alpha=1.1)


def test_alpha0_is_noop_matches_corrupted_baseline_logits():
    model, cfg = _make_tiny()
    clean, corrupt, P = _make_clean_corrupt(cfg, T=12, changed_pos=3)

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    with torch.no_grad():
        _ = model(corrupt)  # baseline
        base_last = model.last_logits.clone()

    with torch.no_grad():
        _ = model(corrupt, layer_to_patch=0, position_to_patch=P, patch_alpha=0.0)
        patched_last = model.last_logits.clone()

    assert torch.allclose(base_last, patched_last, rtol=1e-6, atol=1e-7)
    assert model.last_patch_alpha == 0.0


def test_alpha1_sets_activation_equal_to_clean_cache_at_that_cell():
    model, cfg = _make_tiny()
    clean, corrupt, P = _make_clean_corrupt(cfg, T=12, changed_pos=3)
    L = 0

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    with torch.no_grad():
        _ = model(corrupt, record_activations=True, layer_to_patch=L, position_to_patch=P, patch_alpha=1.0)

    patched_acts = model.last_activations
    assert patched_acts is not None
    assert torch.allclose(
        patched_acts[L][P],
        model.clean_activations[L][P],
        rtol=1e-5,
        atol=1e-6,
    )
    assert model.last_patch == (L, P)
    assert model.last_patch_source == (L, P)
    assert model.last_patch_alpha == 1.0


def test_alpha_half_is_exact_convex_combination_of_clean_and_corrupted_vectors():
    model, cfg = _make_tiny()
    clean, corrupt, P = _make_clean_corrupt(cfg, T=12, changed_pos=3)
    L = 0
    alpha = 0.5

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    # Record corrupted activations (baseline)
    with torch.no_grad():
        _ = model(corrupt, record_activations=True)
    base_acts = model.last_activations
    assert base_acts is not None

    # Patched run with alpha=0.5
    with torch.no_grad():
        _ = model(corrupt, record_activations=True, layer_to_patch=L, position_to_patch=P, patch_alpha=alpha)
    patched_acts = model.last_activations
    assert patched_acts is not None

    expected = (alpha * model.clean_activations[L][P]) + ((1.0 - alpha) * base_acts[L][P])
    assert torch.allclose(patched_acts[L][P], expected, rtol=1e-5, atol=1e-6)
    assert model.last_patch_alpha == alpha


def test_wrong_source_plus_interpolation_uses_clean_source_in_mixture():
    model, cfg = _make_tiny()
    clean, corrupt, P = _make_clean_corrupt(cfg, T=12, changed_pos=3)
    L = 0
    srcP = 0
    alpha = 0.25

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    with torch.no_grad():
        _ = model(corrupt, record_activations=True)
    base_acts = model.last_activations
    assert base_acts is not None

    with torch.no_grad():
        _ = model(
            corrupt,
            record_activations=True,
            layer_to_patch=L,
            position_to_patch=P,
            source_layer=L,
            source_position=srcP,
            patch_alpha=alpha,
        )
    patched_acts = model.last_activations
    assert patched_acts is not None

    expected = (alpha * model.clean_activations[L][srcP]) + ((1.0 - alpha) * base_acts[L][P])
    assert torch.allclose(patched_acts[L][P], expected, rtol=1e-5, atol=1e-6)

    assert model.last_patch == (L, P)
    assert model.last_patch_source == (L, srcP)
    assert model.last_patch_alpha == alpha
