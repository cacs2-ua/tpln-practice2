from __future__ import annotations

import torch

from mingpt.model import GPT
from mingpt.utils import set_seed


def main() -> None:
    set_seed(123)

    # Tiny model (fast, no downloads)
    cfg = GPT.get_default_config()
    cfg.model_type = "gpt-nano"
    cfg.vocab_size = 1000
    cfg.block_size = 64
    model = GPT(cfg).eval()

    # Clean vs corrupted differ by one token id at position 3
    T = 10
    clean = torch.randint(0, cfg.vocab_size, (1, T), dtype=torch.long)
    corrupt = clean.clone()
    corrupt[0, 3] = (corrupt[0, 3] + 1) % cfg.vocab_size  # force one-token change

    # 1) Cache clean activations
    with torch.no_grad():
        _ = model(clean, cache_activations=True, overwrite_cache=True)

    # 2) Corrupt baseline (record activations)
    with torch.no_grad():
        _ = model(corrupt, record_activations=True)
    acts_corrupt = model.last_activations

    # 3) Corrupt + patch at (layer=2, pos=3)
    L, p = 2, 3
    with torch.no_grad():
        _ = model(corrupt, record_activations=True, apply_patch=True, patch_layer=L, patch_position=p)
    acts_patched = model.last_activations

    # Check: patched activation equals cached clean activation at that location
    a_clean = model.clean_activations[L][p]
    a_patch = acts_patched[L][p]
    a_base = acts_corrupt[L][p]

    print("Patched location (layer,pos):", (L, p))
    print("||clean - patched||:", float(torch.norm(a_clean - a_patch)))
    print("||corrupt - patched||:", float(torch.norm(a_base - a_patch)))

    # Check: other positions at that layer unchanged (isolation)
    diffs = []
    for j in range(T):
        if j == p:
            continue
        diffs.append(float(torch.norm(acts_corrupt[L][j] - acts_patched[L][j])))
    print("Max diff at same layer for other positions (should be ~0):", max(diffs) if diffs else 0.0)

    print("\n✅ If the first norm is ~0 and isolation max diff is ~0, patching is correctly placed and isolated.")


if __name__ == "__main__":
    main()
