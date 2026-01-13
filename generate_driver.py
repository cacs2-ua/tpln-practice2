"""
Section 2 driver skeleton (will be extended in Sections 3+ and especially 5–7).

Right now it only:
- loads GPT-2 small via GPT.from_pretrained('gpt2')
- tokenizes a prompt with BPETokenizer
- runs a single forward pass to confirm logits shape
- runs model.generate to confirm decoding loop works

Later, you'll add:
- control flags passed into GPT.forward (save_activations, patch params, etc.)
"""

from __future__ import annotations

import torch

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
from mingpt.utils import set_seed


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main() -> None:
    set_seed(3407)

    device = get_device()
    print("Device:", device)

    model = GPT.from_pretrained("gpt2")
    model.to(device)
    model.eval()

    bpe = BPETokenizer()
    prompt = "Andrej Karpathy, the Earth representative on"
    idx = bpe(prompt).to(device)  # shape (1, T)

    # forward pass (logits for each position)
    logits, loss = model(idx)
    print("Input shape:", tuple(idx.shape))
    print("Logits shape:", tuple(logits.shape))
    assert logits.ndim == 3, "Expected (B, T, V) logits"
    assert logits.shape[0] == idx.shape[0] and logits.shape[1] == idx.shape[1], "B,T must match input"

    # generate a short continuation (just to prove decoding loop works)
    out_idx = model.generate(idx, max_new_tokens=20, do_sample=True, top_k=40)
    out_text = bpe.decode(out_idx[0].cpu())
    print("\n=== Generated ===")
    print(out_text)


if __name__ == "__main__":
    main()
