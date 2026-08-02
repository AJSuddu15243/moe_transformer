"""
linear.py
=========

The Linear (a.k.a. "fully connected" or "dense" or "affine") layer. This is the
single most reused piece of the whole network: attention's Q/K/V/output
projections, the router, the experts' feed-forward matrices, and the final
vocabulary head are ALL Linear layers. Get this one right and half the backward
pass is done.

THE MATH
--------
A Linear layer applies:

        y = x @ W + b

where, if the input has feature dimension `in` and output dimension `out`:

        x : (..., in)      any number of leading "batch-like" dims
        W : (in, out)      the weight matrix   (learnable)
        b : (out,)         the bias vector     (learnable)
        y : (..., out)

The "(...)" means the SAME matrix multiply is applied independently to every
vector in the batch. We flatten all leading dims to a single N = prod(...),
do one 2-D matmul (N,in)@(in,out) -> (N,out), then reshape back. This handles
2-D inputs (N, in) and 3-D inputs (batch, time, in) with identical code.

THE GRADIENTS (chain rule for a matmul + bias)
----------------------------------------------
Given `dout = d loss / d y` with shape (..., out), we want three things.
Work in the flattened 2-D view: X is (N,in), Y = X@W + b is (N,out).

  * d loss / d W :  Y_{n,o} = sum_i X_{n,i} W_{i,o} + b_o.
        dW_{i,o} = sum_n (dY_{n,o}) * X_{n,i}  =>  dW = X^T @ dY      shape (in,out)

  * d loss / d b :  b_o adds to every row, so its gradient is the sum over rows:
        db_o = sum_n dY_{n,o}                  =>  db = dY.sum(axis=0)  shape (out,)

  * d loss / d X :  the gradient we pass upstream to the previous layer.
        dX_{n,i} = sum_o dY_{n,o} * W_{i,o}    =>  dX = dY @ W^T         shape (N,in)

Notice the pleasing symmetry: the forward pass multiplies by W, the backward
pass for the input multiplies by W^T. That transpose relationship shows up in
every linear operation in deep learning.
"""

from __future__ import annotations
import numpy as np
from module import Module


class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, rng: np.random.Generator,
                 std: float = 0.02, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = bias

        # Initialize W with small random normals. Small so that early outputs
        # aren't huge (which would saturate softmaxes and destabilize training).
        # Shape (in, out) so that x @ W maps in-vectors to out-vectors.
        self.params["W"] = rng.standard_normal((in_dim, out_dim)) * std
        # Gradients live alongside params, same shape, start at zero.
        self.grads["W"] = np.zeros_like(self.params["W"])

        if bias:
            # Biases start at 0: at init the layer is a pure linear map.
            self.params["b"] = np.zeros(out_dim)
            self.grads["b"] = np.zeros_like(self.params["b"])

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Remember the original shape so we can un-flatten in backward().
        self._x_shape = x.shape
        # Collapse every leading dim into one: (..., in) -> (N, in).
        self._x2 = x.reshape(-1, self.in_dim)           # cache for backward (need X for dW)
        y2 = self._x2 @ self.params["W"]                # (N, in) @ (in, out) -> (N, out)
        if self.use_bias:
            y2 = y2 + self.params["b"]                  # broadcast bias across all N rows
        # Restore the leading dims, with the last dim now = out.
        return y2.reshape(*self._x_shape[:-1], self.out_dim)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        # Flatten the upstream gradient the same way we flattened the input.
        d2 = dout.reshape(-1, self.out_dim)             # (N, out)

        # dW = X^T @ dY  -- sums the outer products over the whole batch.
        self.grads["W"] = self._x2.T @ d2               # (in, N) @ (N, out) -> (in, out)
        if self.use_bias:
            # db = column-sum of dY over the batch dimension.
            self.grads["b"] = d2.sum(axis=0)            # (out,)

        # dX = dY @ W^T -- gradient handed back to the previous layer.
        dx2 = d2 @ self.params["W"].T                   # (N, out) @ (out, in) -> (N, in)
        return dx2.reshape(self._x_shape)               # un-flatten to match input shape
