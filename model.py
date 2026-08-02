"""
model.py
========

The full model, assembled from the pieces:

    token ids
       |  (1) Embedding: look up a vector per token + add a position vector
       v
    [ Block 0 ]   attention + MoE with residuals   (block.py)
    [ Block 1 ]
       ...
       |
       v  (2) final LayerNorm
       |
       v  (3) LM head: linear map from d_model -> vocab_size
    logits  (one score per possible next character, at every position)

Plus two conveniences: `generate` (sample text from the trained model) and
`save`/`load` (persist the weights to a .npz file).

(1) THE EMBEDDING LAYER
-----------------------
Two learnable tables:
  * token embedding  tok[V, D] : row v is the vector that represents character v.
  * position embedding pos[block_size, D] : row t is a vector that says "I am at
    position t". Attention itself is order-agnostic, so we ADD position vectors
    to give the model a sense of sequence order.
The embedding of position t of a token with id `v` is  tok[v] + pos[t].

Backprop into a lookup table is a "scatter-add": each output row's gradient is
added back onto the table row(s) that produced it (a token id may appear many
times in a batch, so np.add.at accumulates all of their gradients).
"""

from __future__ import annotations
from typing import List
import numpy as np

from module import Module
from linear import Linear
from layernorm import LayerNorm
from block import Block
from attention import softmax


class Embedding(Module):
    """Token + learned positional embeddings."""

    def __init__(self, cfg, rng: np.random.Generator):
        super().__init__()
        self.d_model = cfg.d_model
        # tok: one vector per vocabulary entry. pos: one vector per position.
        self.params["tok"] = rng.standard_normal((cfg.vocab_size, cfg.d_model)) * cfg.init_std
        self.params["pos"] = rng.standard_normal((cfg.block_size, cfg.d_model)) * cfg.init_std
        self.grads["tok"] = np.zeros_like(self.params["tok"])
        self.grads["pos"] = np.zeros_like(self.params["pos"])

    def forward(self, idx: np.ndarray) -> np.ndarray:
        B, T = idx.shape
        self._idx = idx
        self._T = T
        # Fancy-index the token table: tok[idx] has shape (B, T, D).
        tok = self.params["tok"][idx]                       # (B, T, D)
        # Add the first T position vectors, broadcast across the batch.
        pos = self.params["pos"][:T]                        # (T, D)
        return tok + pos[None, :, :]                        # (B, T, D)

    def backward(self, dout: np.ndarray) -> None:
        B, T, D = dout.shape
        # Position grad: sum over the batch (each position row was reused B times).
        dpos = np.zeros_like(self.params["pos"])
        dpos[:T] = dout.sum(axis=0)                         # (T, D)
        self.grads["pos"] = dpos

        # Token grad: scatter-add each position's gradient onto its token row.
        # add.at handles repeated indices correctly (accumulates, not overwrites).
        dtok = np.zeros_like(self.params["tok"])
        np.add.at(dtok, self._idx.reshape(-1), dout.reshape(-1, D))
        self.grads["tok"] = dtok
        # Embedding is the first layer, so there is no input gradient to return.


class MoETransformer(Module):
    def __init__(self, cfg, rng: np.random.Generator):
        super().__init__()
        self.cfg = cfg

        self.embed = self.register("embed", Embedding(cfg, rng))
        # The stack of transformer blocks.
        self.blocks: List[Block] = []
        for i in range(cfg.n_layers):
            b = self.register(f"block{i}", Block(cfg, rng))
            self.blocks.append(b)
        self.ln_f = self.register("ln_f", LayerNorm(cfg.d_model))
        # The language-model head: project the final vector to vocab-size scores.
        # (One could TIE this weight to the token embedding to save params; we
        #  keep it separate so the backward pass stays simple to read.)
        self.lm_head = self.register("lm_head", Linear(cfg.d_model, cfg.vocab_size, rng, std=cfg.init_std))

        # Total auxiliary (load-balancing) loss, summed over all MoE layers.
        self.aux_loss = 0.0

    def forward(self, idx: np.ndarray) -> np.ndarray:
        x = self.embed.forward(idx)                 # (B, T, D)
        for b in self.blocks:
            x = b.forward(x)                        # (B, T, D)
        x = self.ln_f.forward(x)                    # (B, T, D)
        logits = self.lm_head.forward(x)            # (B, T, V)

        # Collect the load-balancing aux losses produced during this forward.
        self.aux_loss = float(sum(b.moe.aux for b in self.blocks))
        return logits

    def backward(self, dlogits: np.ndarray) -> None:
        # Walk backwards through the same layers, in reverse order.
        dx = self.lm_head.backward(dlogits)
        dx = self.ln_f.backward(dx)
        for b in reversed(self.blocks):
            dx = b.backward(dx)
        self.embed.backward(dx)                     # first layer: no return

    # -- text generation ------------------------------------------------------
    def generate(self, idx: np.ndarray, max_new_tokens: int, rng: np.random.Generator,
                 temperature: float = 1.0, top_k: int | None = None) -> np.ndarray:
        """Autoregressively sample new tokens.

        Args:
            idx: (1, t) array of seed token ids to continue from.
            temperature: >1 = more random/creative, <1 = more greedy/confident.
            top_k: if set, sample only from the k most likely next characters.
        Feeds the current sequence in, reads the last position's logits, converts
        them to a probability distribution, samples one token, appends it, and
        repeats. This is exactly how the model "writes".
        """
        block_size = self.cfg.block_size
        for _ in range(max_new_tokens):
            # Never feed more than block_size tokens (the model's context window).
            idx_cond = idx[:, -block_size:]
            logits = self.forward(idx_cond)                 # (1, t, V)
            logits = logits[:, -1, :] / temperature         # (1, V) last step only

            if top_k is not None:
                # Zero out everything except the top_k logits by setting them to -inf.
                kth = np.sort(logits, axis=-1)[:, -top_k][:, None]
                logits = np.where(logits < kth, -1e9, logits)

            probs = softmax(logits, axis=-1)[0]             # (V,)
            next_id = rng.choice(len(probs), p=probs)       # sample from the distribution
            idx = np.concatenate([idx, np.array([[next_id]])], axis=1)
        return idx

    # -- persistence ----------------------------------------------------------
    def save(self, path: str) -> None:
        """Save every parameter into a single .npz, keyed by its dotted name."""
        flat = {name: p for name, p, _ in self.param_items()}
        np.savez(path, **flat)

    def load(self, path: str) -> None:
        """Load parameters saved by save(), copying them IN PLACE so the model's
        arrays (and any optimizer state keyed by name) stay valid."""
        data = np.load(path)
        for name, p, _ in self.param_items():
            p[...] = data[name]
