from __future__ import annotations

import torch
from mingpt.model import GPT
from mingpt.utils import set_seed

@torch.no_grad()
def main():
    set_seed(123)

    # tiny model (fast, no downloads)
    cfg = GPT.get_default_config()
    cfg.model_type = "gpt-nano"
    cfg.vocab_size = 200
    cfg.block_size = 32

    model = GPT(cfg).eval()

    T = 12
    clean = torch.randint(0, cfg.vocab_size, (1, T), dtype=torch.long)
    corrupt = clean.clone()
    corrupt[0, 3] = (corrupt[0, 3] + 1) % cfg.vocab_size  # minimal corruption

    # cache clean activations
    _ = model(clean, cache_activations=True, overwrite_cache=True)
    print("Cached clean activations:",
          len(model.clean_activations), "layers x", len(model.clean_activations[0]), "positions")

    # corrupted baseline (no patch)
    _ = model(corrupt)
    base_last = model.last_logits.clone()

    # one patched run (layer=0, pos=3)
    _ = model(corrupt, layer_to_patch=0, position_to_patch=3)
    patched_last = model.last_logits.clone()

    print("Patch applied:", model.last_patch)
    print("Logits changed vs baseline? ", (not torch.allclose(base_last, patched_last)))

if __name__ == "__main__":
    main()
