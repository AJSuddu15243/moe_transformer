r"""
activations.py
==============

The nonlinearity used inside each expert's feed-forward network: GELU.

WHY A NONLINEARITY AT ALL?
--------------------------
Stacking linear layers is pointless: W2 @ (W1 @ x) is just (W2 @ W1) @ x, i.e.
still ONE linear map. To learn curved, interesting functions we must insert a
nonlinear function between the linear layers. That is the activation function.

WHY GELU?
---------
GELU ("Gaussian Error Linear Unit") is the smooth activation used by GPT/BERT and
most modern transformers. Intuitively it is a "soft ReLU": instead of hard-
gating a value to 0 when it is negative, it multiplies the value by how likely a
standard-normal random variable is to be below it. Small negatives are softly
shrunk rather than chopped off, which gives smoother gradients.

We use the well-known tanh approximation (what GPT-2 uses), which is exact
enough and, crucially, is expressible in PURE NUMPY (the exact GELU needs erf,
which base numpy doesn't provide):

        gelu(x) = 0.5 * x * (1 + tanh( c * (x + 0.044715 * x^3) ))
        with c = sqrt(2/pi) ~= 0.7978845608

THE DERIVATIVE (needed for backprop)
------------------------------------
Let  u = c * (x + 0.044715 x^3)   and   t = tanh(u).
Then gelu = 0.5 x (1 + t), and by the product + chain rule:

        du/dx = c * (1 + 3 * 0.044715 * x^2)
        dt/dx = (1 - t^2) * du/dx            # since d/dx tanh(u) = (1 - tanh^2 u) u'
        d gelu/dx = 0.5 * (1 + t)  +  0.5 * x * (1 - t^2) * du/dx
                    \___________/     \_______________________/
                    derivative of        derivative of the
                    the "0.5 x" factor   "(1+t)" factor, times x

We verify this derivative numerically in gradcheck.py, so you can trust it.
"""

from __future__ import annotations
import numpy as np

# Precompute the constant c = sqrt(2/pi). math is fine here (a plain float).
_C = np.sqrt(2.0 / np.pi)
_A = 0.044715


def gelu_forward(x: np.ndarray):
    """Return (y, cache). `cache` holds what backward needs (here: x and t)."""
    u = _C * (x + _A * x ** 3)          # the inner pre-tanh argument
    t = np.tanh(u)                       # squashed to (-1, 1)
    y = 0.5 * x * (1.0 + t)              # the GELU output
    return y, (x, t)


def gelu_backward(dout: np.ndarray, cache) -> np.ndarray:
    """Given d loss / d y, return d loss / d x."""
    x, t = cache
    du_dx = _C * (1.0 + 3.0 * _A * x ** 2)          # du/dx
    dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du_dx  # d y / d x
    # Chain rule: multiply the local derivative by the upstream gradient.
    return dout * dgelu_dx
