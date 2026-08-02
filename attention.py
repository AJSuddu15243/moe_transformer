"""
attention.py
============

Causal multi-head self-attention -- the mechanism that lets each position mix in
information from earlier positions. This is the "communication" half of a
transformer block (the MoE is the "computation" half).

THE IDEA IN ONE SENTENCE
------------------------
For every token, build a QUERY ("what am I looking for?"), compare it against the
KEY of every earlier token ("what do I offer?"), turn those comparison scores
into attention weights with a softmax, and use them to take a weighted average of
the earlier tokens' VALUES ("what information to actually copy"). That weighted
average is what the token carries forward.

WHY "MULTI-HEAD"?
-----------------
One attention operation can only average things one way. We split the d_model
vector into `n_heads` smaller vectors and run attention independently in each
head, so the model can attend to several kinds of relationships simultaneously
(e.g. one head tracks the previous character, another tracks the start of a word).
The heads' outputs are concatenated back to d_model and mixed by a final linear.

WHY "CAUSAL"?
-------------
We are predicting the NEXT token, so a position must not peek at future tokens
(that would be cheating -- the answer would leak in). We enforce this with a
causal mask that sets the attention score to -infinity for any (query position i,
key position j) with j > i, so softmax gives those future positions weight 0.

SHAPES (B=batch, T=time/positions, D=d_model, H=n_heads, d=D/H=head_dim)
------------------------------------------------------------------------
    input x                : (B, T, D)
    Q, K, V after project  : (B, T, D)  then reshaped to (B, H, T, d)
    scores = QK^T / sqrt(d): (B, H, T, T)
    attn = softmax(scores) : (B, H, T, T)   (row i sums to 1 over allowed keys)
    context = attn @ V     : (B, H, T, d)  -> merge heads -> (B, T, D)
    output                 : (B, T, D)  after the final linear projection

THE MATH, PER HEAD
------------------
        scores = (Q @ K^T) * scale,   scale = 1/sqrt(d)   # scaled dot product
        scores = scores + causal_mask                      # -inf above diagonal
        attn   = softmax(scores)  over the last axis       # weights sum to 1
        context = attn @ V                                 # weighted avg of values

The 1/sqrt(d) scaling keeps the dot products from growing with d (which would
push softmax into a near-one-hot, low-gradient regime).

BACKWARD PASS -- the only new ingredient is the softmax Jacobian:
    if p = softmax(s) (a probability row) and we know dL/dp, then
        dL/ds = p * (dL/dp - sum_j(dL/dp_j * p_j))
    i.e. "multiply by p, then subtract the p-weighted average". Everything else
    is matmul transposes (see linear.py). Derived step by step inline below and
    verified in gradcheck.py.
"""

from __future__ import annotations
import numpy as np
from module import Module
from linear import Linear


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`.

    softmax(x)_i = exp(x_i) / sum_j exp(x_j). We subtract the max first so the
    largest exponent is exp(0)=1, preventing overflow. Subtracting a constant
    from every element does not change the result (it cancels top and bottom).
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class MultiHeadSelfAttention(Module):
    def __init__(self, cfg, rng: np.random.Generator):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_model = cfg.d_model
        self.d_head = cfg.head_dim()
        self.scale = 1.0 / np.sqrt(self.d_head)   # the 1/sqrt(d) factor

        # Four Linear projections. Q, K, V map the input into query/key/value
        # spaces; Wo mixes the concatenated heads back into the residual stream.
        # We register them as children so their params are collected/optimized.
        self.Wq = self.register("Wq", Linear(self.d_model, self.d_model, rng, std=cfg.init_std))
        self.Wk = self.register("Wk", Linear(self.d_model, self.d_model, rng, std=cfg.init_std))
        self.Wv = self.register("Wv", Linear(self.d_model, self.d_model, rng, std=cfg.init_std))
        self.Wo = self.register("Wo", Linear(self.d_model, self.d_model, rng, std=cfg.init_std))

    # -- helpers to split/merge heads ----------------------------------------
    def _split_heads(self, t: np.ndarray) -> np.ndarray:
        # (B, T, D) -> (B, H, T, d). We view D as (H, d) then move H next to batch
        # so that the last two axes (T, d) are what attention operates on.
        B, T, D = t.shape
        t = t.reshape(B, T, self.n_heads, self.d_head)   # split feature dim
        return t.transpose(0, 2, 1, 3)                    # (B, H, T, d)

    def _merge_heads(self, t: np.ndarray) -> np.ndarray:
        # (B, H, T, d) -> (B, T, D). The exact inverse of _split_heads.
        B, H, T, d = t.shape
        t = t.transpose(0, 2, 1, 3)                       # (B, T, H, d)
        return t.reshape(B, T, H * d)                     # concat heads -> D

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape

        # 1) Project inputs to queries, keys, values, then split into heads.
        q = self._split_heads(self.Wq.forward(x))         # (B, H, T, d)
        k = self._split_heads(self.Wk.forward(x))         # (B, H, T, d)
        v = self._split_heads(self.Wv.forward(x))         # (B, H, T, d)

        # 2) Scaled dot-product scores: how much each query matches each key.
        #    k.transpose(...) swaps its last two axes -> (B, H, d, T) so the
        #    matmul contracts over d, giving (B, H, T, T).
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale   # (B, H, T, T)

        # 3) Causal mask: forbid attending to the future (key index > query index).
        #    We build a (T, T) additive mask of 0 (allowed) and -1e9 (forbidden).
        #    np.triu(..., k=1) selects strictly-upper-triangular = future positions.
        causal = np.triu(np.ones((T, T), dtype=bool), k=1)     # True where j > i
        scores = np.where(causal, -1e9, scores)               # push future to -inf

        # 4) Softmax over the key axis -> attention weights that sum to 1 per query.
        attn = softmax(scores, axis=-1)                       # (B, H, T, T)

        # 5) Weighted sum of the values.
        context = attn @ v                                     # (B, H, T, d)

        # 6) Merge heads and apply the output projection.
        context_merged = self._merge_heads(context)          # (B, T, D)
        out = self.Wo.forward(context_merged)                # (B, T, D)

        # Cache what backward needs. (q,k,v,attn are the nonlinear pieces.)
        self._cache = (q, k, v, attn, T)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        q, k, v, attn, T = self._cache

        # Reverse step 6: gradient through the output projection, then un-merge
        # heads to get the gradient w.r.t. `context` (B, H, T, d).
        d_context_merged = self.Wo.backward(dout)             # (B, T, D)
        d_context = self._split_heads(d_context_merged)       # (B, H, T, d)

        # Reverse step 5: context = attn @ v.
        #   d attn = d_context @ v^T   (contract over the d axis)
        #   d v    = attn^T @ d_context
        d_attn = d_context @ v.transpose(0, 1, 3, 2)          # (B, H, T, T)
        dv = attn.transpose(0, 1, 3, 2) @ d_context           # (B, H, T, d)

        # Reverse step 4: the softmax Jacobian (per query row).
        #   d scores = attn * (d attn - sum_j(d attn_j * attn_j))
        # The subtracted term is the attn-weighted average of d_attn over keys.
        weighted = np.sum(d_attn * attn, axis=-1, keepdims=True)   # (B, H, T, 1)
        d_scores = attn * (d_attn - weighted)                     # (B, H, T, T)

        # Reverse step 3: the mask just zeroed forbidden entries (attn=0 there),
        # so no parameter grad flows from it -- nothing to do.

        # Reverse step 2: scores = (q @ k^T) * scale.
        d_scores = d_scores * self.scale                      # undo the scaling
        #   d q = d_scores @ k        (since scores_ij = sum_d q_id k_jd)
        #   d k = d_scores^T @ q
        dq = d_scores @ k                                     # (B, H, T, d)
        dk = d_scores.transpose(0, 1, 3, 2) @ q               # (B, H, T, d)

        # Reverse step 1: merge each head-gradient back to (B, T, D) and push it
        # through the corresponding projection. Each projection saw the SAME
        # input x, so the input gradients from Q, K, V all add up.
        dx = self.Wq.backward(self._merge_heads(dq))
        dx = dx + self.Wk.backward(self._merge_heads(dk))
        dx = dx + self.Wv.backward(self._merge_heads(dv))
        return dx
