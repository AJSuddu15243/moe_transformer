"""
gradcheck.py
============

The single most valuable file for TRUSTING everything else. It numerically
verifies our hand-derived gradients using the definition of a derivative.

THE IDEA: FINITE DIFFERENCES
----------------------------
The derivative of the loss w.r.t. one parameter is, by definition, how much the
loss changes per tiny change in that parameter. We can estimate it WITHOUT any
calculus using the "central difference":

        d loss / d theta  ~=  ( loss(theta + eps) - loss(theta - eps) ) / (2*eps)

for a very small eps. If our analytic backward-pass gradient matches this
numeric estimate for randomly chosen parameters throughout the network, then our
entire chain of derivations (linear, layernorm, gelu, attention softmax, router
softmax, renormalized gating, load-balancing aux, cross-entropy, residuals) is
correct. This is how you debug a from-scratch autograd.

WHY top_k == n_experts HERE
---------------------------
Top-k routing contains a NON-differentiable step: "pick the k highest experts".
When you nudge a router weight, a token can suddenly switch which experts it
picks -- a jump. Right at such a boundary, finite differences and analytic
gradients disagree (the function has a kink). To verify the smooth math cleanly,
we set top_k = n_experts so ALL experts are always selected: no jumps, every
operation differentiable, and the gating/renormalization/aux code paths still
run in full. (Away from those rare boundaries, top_k < n_experts matches too.)

Run:  python gradcheck.py
Expect every parameter group to report a tiny relative error (~1e-6 or smaller).
"""

from __future__ import annotations
import numpy as np

from config import Config
from model import MoETransformer
from losses import cross_entropy


def run_gradcheck():
    # A deliberately tiny model so the whole check is fast and every code path
    # (attention, MoE, residuals, embedding) is still exercised.
    cfg = Config(
        d_model=16, n_heads=2, n_layers=2,
        n_experts=3, top_k=3,            # top_k == n_experts => fully smooth (see header)
        d_ff=16, block_size=6,
        aux_coeff=0.5,                   # nonzero so the aux-loss gradient is tested too
        vocab_size=11,
    )
    rng = np.random.default_rng(0)
    model = MoETransformer(cfg, rng)

    # Random toy batch of token ids and next-token targets.
    B, T = 2, cfg.block_size
    idx = rng.integers(0, cfg.vocab_size, size=(B, T))
    targets = rng.integers(0, cfg.vocab_size, size=(B, T))

    def total_loss():
        """Forward pass -> scalar loss = cross-entropy + load-balancing aux."""
        logits = model.forward(idx)
        ce, dlogits = cross_entropy(logits, targets)
        return ce + model.aux_loss, dlogits

    # --- analytic gradients (our hand-derived backward pass) ---
    loss0, dlogits = total_loss()
    model.backward(dlogits)
    # Snapshot the analytic grads NOW, because the numeric perturbations below
    # will re-run forward/backward and overwrite the grad arrays.
    analytic = {name: g.copy() for name, _, g in model.param_items()}

    print(f"initial loss = {loss0:.6f}")
    print(f"{'parameter':<28}{'max rel err':>14}{'checked':>10}")
    print("-" * 52)

    eps = 1e-5
    rng_pick = np.random.default_rng(123)
    worst_overall = 0.0

    for name, p, _ in model.param_items():
        # Choose a handful of random entries in this parameter to test (checking
        # every entry would be slow; a random sample is a strong signal).
        n_check = min(8, p.size)
        flat_idx = rng_pick.choice(p.size, size=n_check, replace=False)

        worst = 0.0
        for fi in flat_idx:
            orig = p.flat[fi]

            p.flat[fi] = orig + eps
            lp, _ = total_loss()             # loss(theta + eps)
            p.flat[fi] = orig - eps
            lm, _ = total_loss()             # loss(theta - eps)
            p.flat[fi] = orig               # restore

            numeric = (lp - lm) / (2 * eps)
            exact = analytic[name].flat[fi]
            # Floor the denominator so entries whose true gradient is ~0 don't
            # produce a misleadingly huge "relative" error from tiny absolute noise.
            denom = max(1e-6, abs(numeric) + abs(exact))
            rel = abs(numeric - exact) / denom
            worst = max(worst, rel)

        worst_overall = max(worst_overall, worst)
        # 1e-3 is the standard pass bar for a float64 central-difference check on
        # a deep net: real derivation bugs show errors ~1e-1 or larger, while
        # occasional near-cancellation entries can sit around 1e-4 for numerical
        # reasons alone. Anything under 1e-3 means the formula is right.
        flag = "ok" if worst < 1e-3 else "** CHECK **"
        print(f"{name:<28}{worst:>14.2e}{n_check:>10}   {flag}")

    print("-" * 52)
    status = "PASSED" if worst_overall < 1e-3 else "FAILED"
    print(f"worst relative error over all checked params: {worst_overall:.2e}  ->  {status}")


if __name__ == "__main__":
    run_gradcheck()
