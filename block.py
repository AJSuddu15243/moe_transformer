"""
block.py
========

One transformer BLOCK = attention (communication) + MoE (computation), each
wrapped in a LayerNorm and a residual connection. Stacking N of these blocks is
the whole transformer body.

THE PRE-NORM RESIDUAL DESIGN
----------------------------
We use the "pre-norm" arrangement (as in GPT-2 and later): normalize FIRST, then
apply the sublayer, then ADD the result back to the input. In formulas:

        h   = x + Attention( LayerNorm1(x) )
        out = h + MoE(       LayerNorm2(h) )

Two ideas are doing the heavy lifting here:

1. RESIDUAL CONNECTIONS ("x + ..."). Each sublayer only has to learn a *change*
   to add to its input, not reconstruct the whole signal. This creates a
   gradient "highway": during backprop, because out = h + f(h), the gradient
   w.r.t. h is (dout * 1) + (grad through f). That "* 1" path lets gradients
   reach early layers undiminished, which is what makes deep nets trainable.

2. PRE-NORM. Normalizing the sublayer's INPUT (rather than its output) keeps the
   residual stream itself un-normalized and clean, which empirically trains more
   stably and needs no learning-rate warmup tricks to get going.

BACKWARD THROUGH A RESIDUAL
---------------------------
For out = h + sublayer(norm(h)), the incoming gradient `dout` splits and travels
BOTH paths, then recombines:

        d h = dout                           (from the "+ h" identity path)
            + norm.backward( sublayer.backward( dout ) )   (through the sublayer)

We just apply that twice, once per sublayer, walking backwards.
"""

from __future__ import annotations
import numpy as np
from module import Module
from layernorm import LayerNorm
from attention import MultiHeadSelfAttention
from moe import MoE


class Block(Module):
    def __init__(self, cfg, rng: np.random.Generator):
        super().__init__()
        # Two LayerNorms (one before attention, one before the MoE) and the two
        # sublayers. Registered so their parameters are collected for training.
        self.ln1 = self.register("ln1", LayerNorm(cfg.d_model))
        self.attn = self.register("attn", MultiHeadSelfAttention(cfg, rng))
        self.ln2 = self.register("ln2", LayerNorm(cfg.d_model))
        self.moe = self.register("moe", MoE(cfg, rng))

    def forward(self, x: np.ndarray) -> np.ndarray:
        # --- sublayer 1: attention with pre-norm + residual ---
        a = self.ln1.forward(x)                 # normalize the input
        attn_out = self.attn.forward(a)         # let tokens communicate
        h = x + attn_out                        # residual add

        # --- sublayer 2: MoE with pre-norm + residual ---
        b = self.ln2.forward(h)                 # normalize again
        moe_out = self.moe.forward(b)           # per-token expert computation
        out = h + moe_out                       # residual add

        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Reverse of: out = h + moe(ln2(h))
        d_h = dout.copy()                              # gradient along the "+ h" identity
        d_b = self.moe.backward(dout)                  # grad w.r.t. ln2's output
        d_h = d_h + self.ln2.backward(d_b)             # add grad through the MoE path

        # Reverse of: h = x + attn(ln1(x))
        d_x = d_h.copy()                               # gradient along the "+ x" identity
        d_a = self.attn.backward(d_h)                  # grad w.r.t. ln1's output
        d_x = d_x + self.ln1.backward(d_a)             # add grad through the attention path

        return d_x
