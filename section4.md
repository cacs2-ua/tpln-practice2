## 4) Experimental Design: Clean/Corrupted Pair + Hypothesis

**Clean prompt:**
`In Python, the keyword to define a function is`

**Corrupted prompt:**
`In JavaScript, the keyword to define a function is`

**Tokenization constraint:** both prompts have **10** BPE tokens and differ at exactly **one** token position: **1**.

**Token-by-token comparison (diff highlighted):**

| pos | clean_id | clean_tok | corrupt_id | corrupt_tok | diff? |
|---:|---:|---|---:|---|:---:|
| 0 | 818 | 'In' | 818 | 'In' |  |
| 1 | 11361 | ' Python' | 11933 | ' JavaScript' | OK |
| 2 | 11 | ',' | 11 | ',' |  |
| 3 | 262 | ' the' | 262 | ' the' |  |
| 4 | 21179 | ' keyword' | 21179 | ' keyword' |  |
| 5 | 284 | ' to' | 284 | ' to' |  |
| 6 | 8160 | ' define' | 8160 | ' define' |  |
| 7 | 257 | ' a' | 257 | ' a' |  |
| 8 | 2163 | ' function' | 2163 | ' function' |  |
| 9 | 318 | ' is' | 318 | ' is' |  |

**Target competing tokens (next-token prediction at the last position):**
- Token A (clean-consistent): ` def`  (token id: 825)
- Token B (corrupted-consistent): ` function`  (token id: 2163)

**Metric used later (matches the handout):**
`logit(Token B) − logit(Token A)` from the last-position logits.

**Hypothesis:**
Hypothesis: The changed token position (position 1) should matter most, so patching activations at this position across early-to-mid layers should strongly restore the clean-consistent continuation. Middle layers are expected to dominate because they often integrate and route the key conditioning fact/entity forward through the residual stream. Late layers may also show a secondary effect near the final token position because they directly refine the next-token logits.
