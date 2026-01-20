import pytest
import torch

from mingpt.model import GPT

import wrong_source_control as wsc


def _make_tiny():
    cfg = GPT.get_default_config()
    cfg.model_type = "gpt-nano"
    cfg.vocab_size = 200
    cfg.block_size = 32
    model = GPT(cfg).eval()
    return model, cfg


def _make_clean_corrupt(cfg, T=12):
    torch.manual_seed(0)
    clean = torch.randint(0, cfg.vocab_size, (1, T), dtype=torch.long)
    corrupt = clean.clone()
    corrupt[0, 3] = (corrupt[0, 3] + 1) % cfg.vocab_size
    return clean, corrupt


def test_forward_accepts_wrong_source_parameters_and_records_source_bookkeeping():
    model, cfg = _make_tiny()
    clean, corrupt = _make_clean_corrupt(cfg, T=12)

    # cache clean
    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    L_target, P_target = 0, 3
    L_source, P_source = 0, 4

    with torch.no_grad():
        _ = model(
            corrupt,
            record_activations=True,
            layer_to_patch=L_target,
            position_to_patch=P_target,
            source_layer=L_source,
            source_position=P_source,
        )

    assert model.last_patch == (L_target, P_target)
    assert hasattr(model, "last_patch_source")
    assert model.last_patch_source == (L_source, P_source)

    # patched activation at (L_target, P_target) must equal clean cache at (L_source, P_source)
    patched_acts = model.last_activations
    assert patched_acts is not None
    assert torch.allclose(
        patched_acts[L_target][P_target],
        model.clean_activations[L_source][P_source],
        rtol=1e-5,
        atol=1e-6,
    )


def test_standard_patch_is_default_when_source_not_provided():
    model, cfg = _make_tiny()
    clean, corrupt = _make_clean_corrupt(cfg, T=10)

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    L, P = 1, 2
    with torch.no_grad():
        _ = model(corrupt, record_activations=True, layer_to_patch=L, position_to_patch=P)

    assert model.last_patch == (L, P)
    assert model.last_patch_source == (L, P)  # default source == target

    patched_acts = model.last_activations
    assert torch.allclose(
        patched_acts[L][P],
        model.clean_activations[L][P],
        rtol=1e-5,
        atol=1e-6,
    )


def test_wrong_source_pairing_rules_enforced():
    model, cfg = _make_tiny()
    clean, corrupt = _make_clean_corrupt(cfg, T=10)

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    with pytest.raises(ValueError):
        with torch.no_grad():
            _ = model(corrupt, layer_to_patch=0, position_to_patch=3, source_layer=0, source_position=None)

    with pytest.raises(ValueError):
        with torch.no_grad():
            _ = model(corrupt, layer_to_patch=0, position_to_patch=3, source_layer=None, source_position=4)


def test_wrong_source_bounds_checked():
    model, cfg = _make_tiny()
    clean, corrupt = _make_clean_corrupt(cfg, T=8)

    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    # source_position out of range
    with pytest.raises(IndexError):
        with torch.no_grad():
            _ = model(corrupt, layer_to_patch=0, position_to_patch=3, source_layer=0, source_position=999)

    # source_layer out of range
    with pytest.raises(IndexError):
        with torch.no_grad():
            _ = model(corrupt, layer_to_patch=0, position_to_patch=3, source_layer=999, source_position=3)


def test_conditions_for_target_respects_boundaries():
    n_layers = 12
    seq_len = 10

    # P=0 => no WS-pos-
    conds = wsc.conditions_for_target(L=5, P=0, n_layers=n_layers, seq_len=seq_len)
    names = {c.name for c in conds}
    assert "WS-pos-" not in names
    assert "WS-pos+" in names

    # P=seq_len-1 => no WS-pos+
    conds = wsc.conditions_for_target(L=5, P=seq_len - 1, n_layers=n_layers, seq_len=seq_len)
    names = {c.name for c in conds}
    assert "WS-pos+" not in names
    assert "WS-pos-" in names

    # L=0 => no WS-layer-
    conds = wsc.conditions_for_target(L=0, P=3, n_layers=n_layers, seq_len=seq_len)
    names = {c.name for c in conds}
    assert "WS-layer-" not in names
    assert "WS-layer+" in names

    # L=n_layers-1 => no WS-layer+
    conds = wsc.conditions_for_target(L=n_layers - 1, P=3, n_layers=n_layers, seq_len=seq_len)
    names = {c.name for c in conds}
    assert "WS-layer+" not in names
    assert "WS-layer-" in names
