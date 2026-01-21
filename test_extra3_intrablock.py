import pytest
import torch

from mingpt.model import GPT
from mingpt.utils import set_seed


def make_tiny_model():
    set_seed(123)
    cfg = GPT.get_default_config()
    cfg.model_type = "gpt-nano"   # tiny, no downloads
    cfg.vocab_size = 101
    cfg.block_size = 32
    m = GPT(cfg).eval()
    return m


def make_pair(vocab_size: int, T: int = 12, changed_pos: int = 3):
    set_seed(999)
    clean = torch.randint(0, vocab_size, (1, T), dtype=torch.long)
    corrupt = clean.clone()
    corrupt[0, changed_pos] = (corrupt[0, changed_pos] + 1) % vocab_size
    return clean, corrupt, changed_pos


def test_caches_exist_for_both_locations_after_clean_cache():
    m = make_tiny_model()
    vocab = m.transformer.wte.num_embeddings
    clean, _, _ = make_pair(vocab)

    _ = m(clean, cache_activations=True, overwrite_cache=True)

    assert m.clean_post_attn_activations is not None
    assert m.clean_post_mlp_activations is not None
    assert m.clean_activations is not None  # backward compatibility (post-MLP)

    n_layers = len(m.transformer.h)
    T = clean.shape[1]
    assert len(m.clean_post_attn_activations) == n_layers
    assert len(m.clean_post_attn_activations[0]) == T
    assert len(m.clean_post_mlp_activations) == n_layers
    assert len(m.clean_post_mlp_activations[0]) == T

    # clean_activations should match post-MLP cache
    assert torch.allclose(m.clean_activations[0][0], m.clean_post_mlp_activations[0][0])


def test_patch_post_attn_changes_logits():
    m = make_tiny_model()
    vocab = m.transformer.wte.num_embeddings
    clean, corrupt, p = make_pair(vocab)

    _ = m(clean, cache_activations=True, overwrite_cache=True)
    _ = m(corrupt)
    base = m.last_logits.clone()

    _ = m(corrupt, layer_to_patch=0, position_to_patch=p, patch_location="post_attn")
    assert m.last_patch_location == "post_attn"
    assert m.last_patch == (0, p)
    assert not torch.allclose(base, m.last_logits)


def test_patch_post_mlp_changes_logits():
    m = make_tiny_model()
    vocab = m.transformer.wte.num_embeddings
    clean, corrupt, p = make_pair(vocab)

    _ = m(clean, cache_activations=True, overwrite_cache=True)
    _ = m(corrupt)
    base = m.last_logits.clone()

    _ = m(corrupt, layer_to_patch=0, position_to_patch=p, patch_location="post_mlp")
    assert m.last_patch_location == "post_mlp"
    assert m.last_patch == (0, p)
    assert not torch.allclose(base, m.last_logits)


def test_post_attn_and_post_mlp_patches_produce_different_outputs_typically():
    m = make_tiny_model()
    vocab = m.transformer.wte.num_embeddings
    clean, corrupt, p = make_pair(vocab)

    _ = m(clean, cache_activations=True, overwrite_cache=True)

    _ = m(corrupt, layer_to_patch=1, position_to_patch=p, patch_location="post_attn")
    out_attn = m.last_logits.clone()

    _ = m(corrupt, layer_to_patch=1, position_to_patch=p, patch_location="post_mlp")
    out_mlp = m.last_logits.clone()

    # In a random network these should differ (very high probability).
    assert not torch.allclose(out_attn, out_mlp)


def test_invalid_patch_location_raises():
    m = make_tiny_model()
    vocab = m.transformer.wte.num_embeddings
    clean, corrupt, p = make_pair(vocab)

    _ = m(clean, cache_activations=True, overwrite_cache=True)

    with pytest.raises(ValueError):
        _ = m(corrupt, layer_to_patch=0, position_to_patch=p, patch_location="after_unicorns")


def test_patch_and_cache_activations_is_forbidden():
    m = make_tiny_model()
    vocab = m.transformer.wte.num_embeddings
    clean, corrupt, p = make_pair(vocab)

    _ = m(clean, cache_activations=True, overwrite_cache=True)

    with pytest.raises(RuntimeError):
        _ = m(
            corrupt,
            layer_to_patch=0,
            position_to_patch=p,
            patch_location="post_mlp",
            cache_activations=True,
        )
