from __future__ import annotations

import torch
from torch.nn import functional as F

from mingpt.model import GPT
from mingpt.bpe import BPETokenizer


def single_token_id(bpe: BPETokenizer, token_str: str) -> int:
    ids = bpe(token_str)[0].tolist()
    if len(ids) != 1:
        raise ValueError(f"{repr(token_str)} is not a single BPE token. Got {len(ids)} ids: {ids}")
    return int(ids[0])


@torch.no_grad()
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    model = GPT.from_pretrained("gpt2").to(device).eval()
    bpe = BPETokenizer()

    clean = "Juan Antonio watched my neural network learn to juggle bananas; he called it wizard science and demanded espresso"
    idx = bpe(clean).to(device)  # (1, T)

    logits, _ = model(idx)  # forward ONCE
    assert model.last_logits is not None, "model.last_logits was not set!"
    print("logits shape:", tuple(logits.shape))
    print("last_logits shape:", tuple(model.last_logits.shape))  # (1, vocab)

    # Top-k next tokens from last_logits
    k = 10
    last = model.last_logits[0]  # (vocab,)
    probs = F.softmax(last, dim=-1)
    top_p, top_i = torch.topk(probs, k)

    print("\n=== Top-k next tokens (from model.last_logits) ===")
    for rank in range(k):
        tid = int(top_i[rank])
        tok = bpe.decode(torch.tensor([tid]))
        print(f"{rank+1:02d}. id={tid:5d} tok={repr(tok):>12} prob={float(top_p[rank]):.4f}")

    # Metric: logit(TokenB) - logit(TokenA)
    token_a = " wizard"
    token_b = " algorithm"
    id_a = single_token_id(bpe, token_a)
    id_b = single_token_id(bpe, token_b)

    logit_a = float(model.last_logits[0, id_a])
    logit_b = float(model.last_logits[0, id_b])
    score = logit_b - logit_a

    print("\n=== Logit-diff metric ===")
    print(f"Token A: {repr(token_a)} id={id_a} logit={logit_a:.4f}")
    print(f"Token B: {repr(token_b)} id={id_b} logit={logit_b:.4f}")
    print(f"score = logit(B) - logit(A) = {score:.4f}")


if __name__ == "__main__":
    main()
