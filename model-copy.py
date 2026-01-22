"""
Full definition of a GPT Language Model, all of it in this single file.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CN()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                # names follow the huggingface naming conventions
                # GPT-1
                'openai-gpt':   dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
                'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
                'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
                # Gophers
                'gopher-44m':   dict(n_layer=8, n_head=16, n_embd=512),
                # (there are a number more...)
                # I made these tiny models up
                'gpt-mini':     dict(n_layer=6, n_head=6, n_embd=192),
                'gpt-micro':    dict(n_layer=4, n_head=4, n_embd=128),
                'gpt-nano':     dict(n_layer=3, n_head=3, n_embd=48),
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

        # --- Mechanistic interpretability instrumentation (Section 5) ---
        # clean cache: list[layer][position] -> Tensor(d_model) for batch element 0
        self.clean_activations: Optional[List[List[torch.Tensor]]] = None
        self.clean_activation_meta: Optional[Dict[str, int]] = None
        # last recorded activations (debug/inspection; does NOT overwrite clean cache)
        self.last_activations: Optional[List[List[torch.Tensor]]] = None


        # last-token logits (Section 6): logits at final prompt position (next-token distribution)
        self.last_logits: Optional[torch.Tensor] = None
        self.last_patch: Optional[Tuple[int, int]] = None
        self.last_patch_source: Optional[Tuple[int, int]] = None   # (source_layer, source_pos)
        self.last_patch_alpha: Optional[float] = None
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    @classmethod
    def from_pretrained(cls, model_type):
        """
        Initialize a pretrained GPT model by copying over the weights
        from a huggingface/transformers checkpoint.
        """
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel

        # create a from-scratch initialized minGPT model
        config = cls.get_default_config()
        config.model_type = model_type
        config.vocab_size = 50257 # openai's model vocabulary
        config.block_size = 1024  # openai's model block_size
        model = GPT(config)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        keys = [k for k in sd_hf if not k.endswith('attn.masked_bias')] # ignore these
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla nn.Linear.
        # this means that we have to transpose these weights when we import them
        assert len(keys) == len([k for k in sd if not k.endswith(".attn.bias")])
        for k in keys:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer


    def clear_clean_activations(self) -> None:
        """Clear the stored clean activation cache (Section 5)."""
        self.clean_activations = None
        self.clean_activation_meta = None
    def forward(
        self,
        idx,
        targets=None,
        *,
        record_activations: bool = False,
        cache_activations: bool = False,
        overwrite_cache: bool = False,
        layer_to_patch: Optional[int] = None,
        position_to_patch: Optional[int] = None,
        # NEW (EXTRA 1): wrong-source patch controls
        source_layer: Optional[int] = None,
        source_position: Optional[int] = None,
        patch_alpha: Optional[float] = None,
    ):
        """
        Forward pass with:
        - optional activation recording (Section 5)
        - optional clean activation caching (Section 5)
        - last-position logits extraction (Section 6)
        - activation patching (Section 7)
        - WRONG-SOURCE patching control (EXTRA 1)

        Activation definition:
        - residual stream output AFTER each transformer block
        - recorded for each token position
        - stored for batch element 0 only
        - deep-copied via detach().clone()

        Patching semantics:
        Target (where we write): (layer_to_patch, position_to_patch) in the corrupted run
        Source (what we copy): (source_layer, source_position) from clean cache

        Standard patch is the special case:
            source_layer == layer_to_patch and source_position == position_to_patch
        """
        # If we're caching, we must record
        record_activations = bool(record_activations or cache_activations)

        # Reset per-run outputs (safe defaults)
        self.last_activations = None
        self.last_logits = None
        self.last_patch = None
        self.last_patch_source = None
        self.last_patch_alpha = None

        # --- Patch argument validation (isolation + safety) ---
        patch_requested = (layer_to_patch is not None) or (position_to_patch is not None)
        source_requested = (source_layer is not None) or (source_position is not None)

        if patch_requested:
            # must provide BOTH target coords
            if (layer_to_patch is None) or (position_to_patch is None):
                raise ValueError("Patching requires BOTH layer_to_patch and position_to_patch (or neither).")

            # if user provides any source coord, must provide BOTH
            if source_requested and ((source_layer is None) or (source_position is None)):
                raise ValueError("Wrong-source patching requires BOTH source_layer and source_position (or neither).")

            # default: standard matching patch if source not provided
            if not source_requested:
                source_layer = layer_to_patch
                source_position = position_to_patch

            # do not allow writing the clean cache during patched runs (clean-cache safety)
            if cache_activations:
                raise RuntimeError(
                    "Refusing to run patching with cache_activations=True. "
                    "Clean cache must only be written during the clean run."
                )
            if overwrite_cache:
                raise RuntimeError(
                    "overwrite_cache=True is not allowed during patching runs. "
                    "Clean cache must remain immutable during corrupted/patched runs."
                )

            # must already have a clean cache
            if self.clean_activations is None:
                raise RuntimeError(
                    "No clean activation cache found (self.clean_activations is None). "
                    "Run a CLEAN pass first with cache_activations=True."
                )

        patch_alpha_f: float = 1.0
        if patch_requested:
            if patch_alpha is None:
                patch_alpha_f = 1.0
            else:
                patch_alpha_f = float(patch_alpha)
            if not (0.0 <= patch_alpha_f <= 1.0):
                raise ValueError(f"patch_alpha must be in [0,1], got {patch_alpha_f}")
        
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)

        # Prepare activation container (optional)
        acts = None
        if record_activations:
            acts = []  # list[layer][pos] -> Tensor(d_model)

        # Extra patch compatibility checks once seq_len is known
        if patch_requested:
            n_layer = len(self.transformer.h)

            # target checks
            if not (0 <= int(layer_to_patch) < n_layer):
                raise IndexError(f"layer_to_patch out of range: {layer_to_patch} (valid: 0..{n_layer-1})")
            if not (0 <= int(position_to_patch) < t):
                raise IndexError(f"position_to_patch out of range: {position_to_patch} (valid: 0..{t-1})")

            # source checks
            if not (0 <= int(source_layer) < n_layer):
                raise IndexError(f"source_layer out of range: {source_layer} (valid: 0..{n_layer-1})")
            if not (0 <= int(source_position) < t):
                raise IndexError(f"source_position out of range: {source_position} (valid: 0..{t-1})")

            # Check cache dimensions match current run
            if len(self.clean_activations) != n_layer:
                raise RuntimeError(
                    f"Clean cache layer count mismatch: cache has {len(self.clean_activations)} "
                    f"but model has {n_layer} layers."
                )
            cached_t = len(self.clean_activations[0]) if n_layer > 0 else 0
            if cached_t != t:
                raise RuntimeError(
                    f"Sequence length mismatch vs clean cache: clean cache T={cached_t}, current T={t}. "
                    "Clean and corrupted prompts must have identical token length, "
                    "and you must re-cache clean activations if the prompt length changes."
                )

        patch_applied = False

        # --- Transformer block stack ---
        for layer_idx, block in enumerate(self.transformer.h):
            x = block(x)

            # --- Apply patch exactly after the chosen TARGET layer output ---
            if patch_requested and (layer_idx == int(layer_to_patch)):
                clean_vec = self.clean_activations[int(source_layer)][int(source_position)]
                clean_vec = clean_vec.to(device=x.device, dtype=x.dtype)

                if patch_alpha_f <= 0.0:
                    # alpha=0 => no-op (keep corrupted activation)
                    pass
                elif patch_alpha_f >= 1.0:
                    # alpha=1 => standard full patch
                    x[0, int(position_to_patch), :].copy_(clean_vec)
                else:
                    # 0<alpha<1 => convex combination
                    corr_vec = x[0, int(position_to_patch), :].detach().clone()
                    mixed_vec = (patch_alpha_f * clean_vec) + ((1.0 - patch_alpha_f) * corr_vec)
                    x[0, int(position_to_patch), :].copy_(mixed_vec)

                patch_applied = True
                self.last_patch = (int(layer_to_patch), int(position_to_patch))
                self.last_patch_source = (int(source_layer), int(source_position))
                self.last_patch_alpha = patch_alpha_f
            # --- record activations AFTER patching ---
            if record_activations:
                layer_acts = []
                for p in range(t):
                    layer_acts.append(x[0, p, :].detach().clone())
                acts.append(layer_acts)

        if patch_requested and (not patch_applied):
            raise RuntimeError("Patch was requested but not applied (internal logic error).")

        # --- Final norm + LM head ---
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # --- store last-position logits (next-token distribution after the prompt) ---
        self.last_logits = logits[:, -1, :].detach().clone()

        # loss (optional)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        # finalize activation storage
        if record_activations:
            self.last_activations = acts

        # write clean cache ONLY when explicitly requested
        if cache_activations:
            if (self.clean_activations is not None) and (not overwrite_cache):
                raise RuntimeError(
                    "Clean activation cache already exists. "
                    "Pass overwrite_cache=True (or call model.clear_clean_activations()) "
                    "to replace it for a new clean prompt."
                )

            self.clean_activations = acts
            self.clean_activation_meta = {
                "seq_len": int(t),
                "n_layer": int(len(self.transformer.h)),
                "d_model": int(x.shape[-1]),
            }

        return logits, loss




    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # either sample from the distribution or take the most likely element
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx