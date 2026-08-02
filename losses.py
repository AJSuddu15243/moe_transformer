"""
losses.py
=========

Softmax cross-entropy: how we turn the model's raw output scores into a single
number to minimize, and how the gradient of that number w.r.t. the scores has a
famously clean form.

THE SETUP
---------
For each position the model outputs `logits`: one raw score per vocabulary entry
(size V). The true next character is a single class index `target`. We want the
model to put high probability on the true class.

    probs = softmax(logits)                 # scores -> probabilities over V classes
    loss  = -log( probs[target] )           # negative log-likelihood of the truth

Averaged over all N = batch*time positions, this is the cross-entropy loss.
Minimizing it maximizes the probability the model assigns to the correct next
characters. A loss of ln(V) means "random guessing"; lower is better. (For our
vocab, ln(V) is roughly 3.6 -- watch training drop below that fast.)

THE BEAUTIFUL GRADIENT
----------------------
Cross-entropy composed with softmax has the simplest gradient in deep learning:

        d loss / d logits = (probs - onehot(target)) / N

That is: "predicted probabilities minus the one-hot truth". If the model already
puts prob 1 on the right class, probs - onehot = 0 -> no gradient. Otherwise the
gradient pushes the correct logit up and the others down, in exact proportion to
how wrong the probabilities are. The messy softmax and log derivatives cancel
perfectly when combined -- which is exactly why they are always fused into one
"softmax cross-entropy" op. We divide by N because we average the per-token
losses (so the gradient is an average too).
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from attention import softmax


def cross_entropy(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Args:
        logits:  (B, T, V) raw scores over the vocabulary at every position
        targets: (B, T)    integer id of the true next token at every position
    Returns:
        loss:    scalar average negative log-likelihood
        dlogits: (B, T, V) gradient d loss / d logits
    """
    B, T, V = logits.shape
    N = B * T

    # Flatten the (B, T) grid of positions into one long list of N examples.
    flat_logits = logits.reshape(N, V)          # (N, V)
    flat_targets = targets.reshape(N)           # (N,)

    # Convert scores to probabilities (numerically stable softmax).
    probs = softmax(flat_logits, axis=-1)       # (N, V)

    # Pick out the probability assigned to the TRUE class at each position.
    # np.arange(N) pairs with flat_targets to index one entry per row.
    correct = probs[np.arange(N), flat_targets]     # (N,)

    # Average negative log-likelihood. Clip to avoid log(0) = -inf if a prob
    # underflows to exactly 0.
    loss = float(-np.mean(np.log(correct + 1e-12)))

    # Gradient: start from probs, subtract 1 from the true-class entry (the
    # one-hot), then average by dividing by N.
    dlogits = probs.copy()
    dlogits[np.arange(N), flat_targets] -= 1.0
    dlogits /= N

    return loss, dlogits.reshape(B, T, V)
