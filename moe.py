"""
moe.py
======

The Mixture-of-Experts layer -- the star of the show. In a vanilla transformer,
the "computation" half of each block is a single feed-forward network (FFN)
applied to every token. MoE replaces that one FFN with MANY expert FFNs plus a
tiny ROUTER that, per token, picks a few experts to actually run.

WHY DO THIS?
------------
Capacity vs. compute. Adding parameters usually costs proportionally more
compute. MoE breaks that link: with `n_experts` experts but only `top_k` of them
active per token, the model holds a LOT of parameters (all experts) while each
token only pays for `top_k` of them. Different experts specialize on different
kinds of tokens. It is "conditional computation": the network chooses its own
compute path per input.

THE FOUR STEPS (per token, feature vector x of size D)
------------------------------------------------------
 1. ROUTE:   logits = x @ W_router                 -> one score per expert (size E)
             probs  = softmax(logits)              -> a distribution over experts
 2. SELECT:  keep the top_k experts by probability; call that set S.
             renormalize their probs to sum to 1:  gate_e = probs_e / sum_{j in S} probs_j
             (renormalization = "given we only use these k, how to weight them")
 3. COMPUTE: run each chosen expert's FFN on x:     h_e = Expert_e(x)
 4. COMBINE: weighted sum of the chosen experts:    y = sum_{e in S} gate_e * h_e

An EXPERT is just a 2-layer FFN:  Expert(x) = fc2( gelu( fc1(x) ) ).

LOAD BALANCING (the auxiliary loss)
-----------------------------------
Left alone, the router tends to collapse -- it discovers a couple of "good"
experts and sends everything to them, wasting the rest. To prevent that we add a
small auxiliary loss that is minimized when tokens are spread EVENLY across
experts (Switch-Transformer style):

        aux = aux_coeff * E * sum_e ( f_e * P_e )

    f_e = fraction of tokens that routed to expert e   (a hard count; treated as
          a constant w.r.t. gradients -- it comes from a non-differentiable top-k)
    P_e = average router probability assigned to expert e over the batch
          (this IS differentiable and is where the gradient acts)

If some expert is overused, both its f_e (many tokens) and P_e (high prob) are
large, so their product is large -> the loss pushes P_e down for that expert,
encouraging balance. Because f_e is a constant here, the gradient is simply:

        d aux / d probs[n, e] = aux_coeff * E * f_e / N        (for every token n)

------------------------------------------------------------------------------
A NOTE ON EFFICIENCY (important for understanding real MoE):
Real MoE implementations DISPATCH: they gather only the tokens each expert needs
and run each expert on its own small batch, so skipped experts cost nothing.
Here, for clarity and correct-by-construction backprop, we instead run EVERY
expert on EVERY token and multiply the un-chosen ones by a gate of 0. The MATH
of routing/gating/backprop is identical; only the compute is wasteful. With a
handful of experts and a tiny model this is perfectly fast, and it keeps the
gradient code short and verifiable. The header of each step notes where the
sparsity would kick in.
------------------------------------------------------------------------------
"""

from __future__ import annotations
import numpy as np
from module import Module
from linear import Linear
from activations import gelu_forward, gelu_backward
from attention import softmax


class Expert(Module):
    """A single expert = a 2-layer feed-forward network with a GELU in between.

        Expert(x) = fc2( gelu( fc1(x) ) )
        fc1: D -> d_ff   (expand)
        fc2: d_ff -> D   (project back)
    """

    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator, std: float):
        super().__init__()
        self.fc1 = self.register("fc1", Linear(d_model, d_ff, rng, std=std))
        self.fc2 = self.register("fc2", Linear(d_ff, d_model, rng, std=std))

    def forward(self, x: np.ndarray) -> np.ndarray:
        a = self.fc1.forward(x)                 # (N, d_ff)  expand
        g, self._gcache = gelu_forward(a)       # (N, d_ff)  nonlinearity
        return self.fc2.forward(g)              # (N, D)     project back

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dg = self.fc2.backward(dout)            # grad into the GELU output
        da = gelu_backward(dg, self._gcache)    # grad through the GELU
        return self.fc1.backward(da)            # grad into the expert's input


class MoE(Module):
    def __init__(self, cfg, rng: np.random.Generator):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.top_k = cfg.top_k
        self.aux_coeff = cfg.aux_coeff
        self.d_model = cfg.d_model

        # The router: a linear map from a token vector to one score per expert.
        self.router = self.register("router", Linear(cfg.d_model, cfg.n_experts, rng, std=cfg.init_std))

        # The experts. Registered with distinct names so their params collect.
        self.experts = []
        for i in range(cfg.n_experts):
            e = self.register(f"expert{i}", Expert(cfg.d_model, cfg.d_ff, rng, std=cfg.init_std))
            self.experts.append(e)

        # After each forward, we stash the auxiliary load-balancing loss here so
        # the model can add it to the reported loss. Its gradient is injected
        # inside backward() automatically (it does not arrive via `dout`).
        self.aux = 0.0

    def forward(self, x: np.ndarray) -> np.ndarray:
        B, T, D = x.shape
        N = B * T
        E = self.n_experts
        xf = x.reshape(N, D)                        # flatten tokens: (N, D)

        # ---- Step 1: ROUTE ----
        logits = self.router.forward(xf)           # (N, E) one score per expert
        probs = softmax(logits, axis=-1)           # (N, E) distribution over experts

        # ---- Step 2: SELECT top_k and renormalize ----
        # argsort ascending; the last top_k columns are the largest-prob experts.
        idx = np.argsort(probs, axis=1)[:, -self.top_k:]         # (N, k) chosen expert ids
        vals = np.take_along_axis(probs, idx, axis=1)           # (N, k) their probs
        denom = vals.sum(axis=1, keepdims=True)                 # (N, 1) normalizer
        gates_k = vals / denom                                  # (N, k) renormalized weights

        # Scatter the per-token gates back into dense (N, E) matrices:
        #   G[n, e] = gate weight if expert e is chosen for token n, else 0
        #   M[n, e] = 1 if chosen, else 0   (the "dispatch mask")
        G = np.zeros((N, E))
        np.put_along_axis(G, idx, gates_k, axis=1)
        M = np.zeros((N, E))
        np.put_along_axis(M, idx, np.ones_like(gates_k), axis=1)

        # ---- Load-balancing statistics (for the aux loss) ----
        f = M.mean(axis=0)                          # (E,) fraction of tokens per expert
        P = probs.mean(axis=0)                      # (E,) mean router prob per expert
        self.aux = self.aux_coeff * E * float(np.sum(f * P))

        # ---- Step 3 + 4: COMPUTE each expert and COMBINE by gate ----
        # (Sparsity note: in a real MoE only tokens with G[:,e] != 0 would run
        #  through expert e. Here we run all tokens through every expert.)
        y = np.zeros((N, D))
        expert_outputs = []
        for e in range(E):
            h_e = self.experts[e].forward(xf)       # (N, D) this expert on all tokens
            expert_outputs.append(h_e)
            y += G[:, e:e + 1] * h_e                # weight by the gate (0 if not chosen)

        # Cache for backward.
        self._cache = (xf.shape, B, T, D, N, E, probs, G, M, denom, f, expert_outputs)
        return y.reshape(B, T, D)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        (xf_shape, B, T, D, N, E, probs, G, M, denom, f, expert_outputs) = self._cache
        dy = dout.reshape(N, D)                     # (N, D) grad w.r.t. combined output

        dxf = np.zeros(xf_shape)                    # accumulate grad w.r.t. MoE input
        dG = np.zeros((N, E))                       # grad w.r.t. the gate matrix

        # ---- reverse Step 3+4: through each expert ----
        for e in range(E):
            h_e = expert_outputs[e]
            # y += G[:,e]*h_e, so:
            #   d h_e   = dy * G[:,e]                (gate scales the expert output)
            #   dG[:,e] = sum_d dy * h_e             (how the combined output moved with the gate)
            d_h_e = dy * G[:, e:e + 1]              # (N, D)
            dG[:, e] = np.sum(dy * h_e, axis=1)     # (N,)
            dxf += self.experts[e].backward(d_h_e)  # grad back through the expert FFN

        # ---- reverse Step 2: through the renormalized top-k gating ----
        # For a token, let S = chosen experts, denom = sum_{j in S} probs_j, and
        # gate_e = probs_e / denom for e in S. Differentiating gate_e w.r.t. the
        # chosen probs and folding in dG gives, for each chosen expert a:
        #     d probs_a = ( dG_a - sum_{e in S} gate_e * dG_e ) / denom
        # and 0 for non-chosen experts (they don't enter the gate formula).
        # G already holds gate_e (and is 0 for non-chosen), M masks to S.
        sum_gate_dG = np.sum(G * dG, axis=1, keepdims=True)     # (N, 1) = sum_{e in S} gate_e dG_e
        d_probs = M * (dG - sum_gate_dG) / denom               # (N, E), nonzero only on S

        # ---- add the auxiliary load-balancing gradient ----
        # d aux / d probs[n,e] = aux_coeff * E * f_e / N, for ALL tokens/experts.
        # (f treated as a constant; it came from the non-differentiable top-k.)
        d_probs = d_probs + (self.aux_coeff * E * f[None, :] / N)

        # ---- reverse Step 1: softmax Jacobian, then the router linear ----
        #   d logits = probs * (d probs - sum_j(d probs_j * probs_j))
        weighted = np.sum(d_probs * probs, axis=1, keepdims=True)   # (N, 1)
        d_logits = probs * (d_probs - weighted)                     # (N, E)
        dxf += self.router.backward(d_logits)                       # (N, D)

        return dxf.reshape(B, T, D)
