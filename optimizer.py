"""
optimizer.py
============

Adam, the optimizer that actually updates the weights using the gradients we
worked so hard to compute. This is the "learning" step: for every parameter,
nudge it downhill on the loss.

PLAIN GRADIENT DESCENT and why we improve on it
-----------------------------------------------
The simplest update is  theta <- theta - lr * grad. Two problems:
  1. A single global learning rate `lr` is wrong for parameters whose gradients
     have very different scales -- some want big steps, some tiny ones.
  2. Noisy per-batch gradients make the path zig-zag.

ADAM fixes both by keeping two running (exponential moving) averages per param:
  * m ("first moment")  ~ the average recent gradient       -> gives momentum,
                                                                 smoothing noise.
  * v ("second moment") ~ the average recent gradient SQUARED-> a per-parameter
                                                                 scale estimate.
The update divides the smoothed gradient by sqrt(v). So parameters with big,
consistent gradients (large v) take smaller, careful steps; parameters with tiny
gradients take relatively larger ones. Every parameter gets its own adaptive
step size -- hence "Adaptive Moment estimation".

THE UPDATE (per parameter, at step t)
-------------------------------------
        m <- beta1 * m + (1 - beta1) * g            # update 1st moment
        v <- beta2 * v + (1 - beta2) * g^2          # update 2nd moment

        m_hat <- m / (1 - beta1^t)                  # bias correction (see below)
        v_hat <- v / (1 - beta2^t)

        theta <- theta - lr * m_hat / (sqrt(v_hat) + eps)

BIAS CORRECTION: m and v start at 0, so early on they are biased toward 0
(there hasn't been enough history to "fill them up"). Dividing by (1 - beta^t)
inflates them to compensate; as t grows, beta^t -> 0 and the correction fades.

WEIGHT DECAY (optional): subtract a little bit of the weight itself each step,
        theta <- theta - lr * wd * theta
which gently pulls weights toward 0 -- a form of regularization. Off by default.
"""

from __future__ import annotations
import numpy as np


class Adam:
    def __init__(self, model, lr: float, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.0):
        self.model = model
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # step counter, drives bias correction

        # One m and one v array per parameter, keyed by the parameter's unique
        # dotted name. Initialized to zeros (no history yet).
        self.m = {}
        self.v = {}
        for name, p, _ in model.param_items():
            self.m[name] = np.zeros_like(p)
            self.v[name] = np.zeros_like(p)

    def step(self) -> None:
        """Apply one Adam update to every parameter, using the gradients that
        the most recent backward() pass stored in the model."""
        self.t += 1
        # Precompute the bias-correction denominators for this step.
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t

        # Re-collect (name, param, grad) fresh so we read THIS step's gradients.
        for name, p, g in self.model.param_items():
            # Optional decoupled weight decay: shrink the weight slightly.
            if self.weight_decay != 0.0:
                p -= self.lr * self.weight_decay * p

            # Update the moving averages of the gradient and its square.
            m = self.m[name]
            v = self.v[name]
            m[...] = self.beta1 * m + (1.0 - self.beta1) * g
            v[...] = self.beta2 * v + (1.0 - self.beta2) * (g * g)

            # Bias-corrected estimates.
            m_hat = m / bc1
            v_hat = v / bc2

            # The adaptive step. `p -= ...` mutates the parameter array in place,
            # so the model immediately sees the update.
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Not strictly needed here: our backward() fully overwrites each grad
        every pass (it assigns, not accumulates), so there is nothing stale to
        clear. Provided for familiarity with the usual training-loop shape."""
        pass
