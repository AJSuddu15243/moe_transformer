"""
layernorm.py
============

Layer Normalization. This is the "keep the numbers sane" layer. It sits before
attention and before the MoE inside every block (the "pre-norm" design).

THE PROBLEM IT SOLVES
---------------------
As signals pass through many layers, the scale of the activation vectors can
drift -- growing or shrinking -- which makes gradients explode or vanish and
training unstable. LayerNorm fixes this by re-centering and re-scaling each
token's feature vector to have mean 0 and variance 1, and THEN letting the model
re-stretch/re-shift it with two learnable vectors (gamma, beta). So the network
keeps full expressive power but starts every layer from a well-conditioned place.

Crucially, LayerNorm normalizes ACROSS THE FEATURE DIMENSION of a single token,
independently per token. It does not mix information across positions or across
the batch (unlike BatchNorm). That independence is what makes it play nicely
with variable-length sequences and autoregressive generation.

THE MATH (for one token's feature vector x of length D)
-------------------------------------------------------
        mu    = (1/D) * sum_j x_j                      # mean over features
        var   = (1/D) * sum_j (x_j - mu)^2             # variance over features
        xhat  = (x - mu) / sqrt(var + eps)             # normalized: mean 0, var 1
        y     = gamma * xhat + beta                    # learnable scale & shift

`gamma` (scale) and `beta` (shift) are length-D learnable vectors, shared across
all tokens. `eps` is a tiny constant so we never divide by zero.

THE BACKWARD PASS
-----------------
gamma and beta are easy (they are just an elementwise multiply and add):

        d gamma = sum over all tokens of (dy * xhat)     # (D,)
        d beta  = sum over all tokens of  dy             # (D,)

The gradient w.r.t. the input x is the subtle part, because mu and var each
depend on ALL of x's entries, so changing one x_j moves the whole normalized
vector. Carefully applying the chain rule through mu, var and the 1/sqrt gives
this compact, well-known closed form (per token, D = feature dim):

        dxhat = dy * gamma
        dx = (1/D) * inv_std * ( D*dxhat
                                 - sum_j dxhat_j
                                 - xhat * sum_j (dxhat_j * xhat_j) )

where inv_std = 1/sqrt(var + eps). Read it as: "take the incoming gradient,
then remove its mean component and its projection onto xhat" -- those two
subtractions are exactly the corrections for how each x_j also nudged mu and var.
We confirm this formula numerically in gradcheck.py.
"""

from __future__ import annotations
import numpy as np
from module import Module


class LayerNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # gamma starts at 1 and beta at 0 => at init, y == xhat (pure normalize).
        self.params["gamma"] = np.ones(dim)
        self.params["beta"] = np.zeros(dim)
        self.grads["gamma"] = np.zeros_like(self.params["gamma"])
        self.grads["beta"] = np.zeros_like(self.params["beta"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x has shape (..., D). We normalize over the LAST axis only.
        mu = x.mean(axis=-1, keepdims=True)                  # (..., 1) per-token mean
        # Use the population variance (divide by D), matching the derivative above.
        var = x.var(axis=-1, keepdims=True)                  # (..., 1) per-token variance
        inv_std = 1.0 / np.sqrt(var + self.eps)              # (..., 1)
        xhat = (x - mu) * inv_std                            # (..., D) normalized

        # Cache the pieces backward() needs. inv_std and xhat are enough.
        self._xhat = xhat
        self._inv_std = inv_std

        return self.params["gamma"] * xhat + self.params["beta"]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        xhat = self._xhat
        inv_std = self._inv_std
        D = self.dim

        # We sum the param grads over every leading axis (all tokens), leaving
        # shape (D,). axis=tuple of all but the last.
        reduce_axes = tuple(range(dout.ndim - 1))
        self.grads["gamma"] = np.sum(dout * xhat, axis=reduce_axes)
        self.grads["beta"] = np.sum(dout, axis=reduce_axes)

        # dxhat = dy * gamma  (chain rule through the final scale).
        dxhat = dout * self.params["gamma"]                  # (..., D)

        # The two "correction" sums, computed per token (over the feature axis).
        sum_dxhat = np.sum(dxhat, axis=-1, keepdims=True)                 # (..., 1)
        sum_dxhat_xhat = np.sum(dxhat * xhat, axis=-1, keepdims=True)     # (..., 1)

        # The compact LayerNorm input-gradient formula (derived in the header).
        dx = (inv_std / D) * (D * dxhat - sum_dxhat - xhat * sum_dxhat_xhat)
        return dx
