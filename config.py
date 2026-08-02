"""
config.py
=========

Every knob of the model and the training run, in ONE place. Keeping the
hyperparameters separate from the code means you can experiment ("what if I use
8 experts instead of 4?") by editing numbers here, without touching any math.

Below each field is a note on what it controls and how it affects size/compute.
The defaults are deliberately TINY so the whole thing trains on a laptop CPU in
pure numpy in a couple of minutes, while still being a *real* Mixture-of-Experts
transformer (multi-head causal attention + top-k routed experts + load balancing
+ Adam + backprop).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Config:
    # ---- model shape --------------------------------------------------------
    # d_model (a.k.a. the "embedding dimension" or "residual stream width"):
    # the length of the vector that represents each token as it flows through
    # the network. Everything -- embeddings, attention, experts -- reads and
    # writes vectors of this width. Bigger = more capacity, more compute.
    d_model: int = 64

    # Number of attention heads. The d_model vector is split into n_heads
    # chunks of size d_model/n_heads, and each head does attention independently.
    # Multiple heads let the model attend to several relationships at once.
    # REQUIREMENT: d_model must be divisible by n_heads.
    n_heads: int = 4

    # Number of stacked transformer blocks (each = attention + MoE). Depth.
    n_layers: int = 2

    # ---- Mixture of Experts -------------------------------------------------
    # Each MoE layer contains `n_experts` independent feed-forward networks.
    # A per-token "router" picks which experts handle each token.
    n_experts: int = 4

    # top_k: how many experts each token is routed to (the "sparse" part).
    # With top_k < n_experts, most experts are skipped for any given token,
    # which is the whole point of MoE: lots of parameters, little compute per
    # token. top_k=2 is the common choice (e.g. Mixtral).
    top_k: int = 2

    # Hidden width INSIDE each expert's feed-forward network. Classic
    # transformers use ~4*d_model; we use 2*d_model to stay small.
    d_ff: int = 128

    # Coefficient on the load-balancing auxiliary loss (see moe.py). This gently
    # pushes the router to use all experts evenly instead of collapsing onto a
    # favorite few. Set to 0.0 to disable.
    aux_coeff: float = 0.01

    # ---- sequence / vocab ---------------------------------------------------
    # block_size = context length = how many previous characters the model may
    # look at when predicting the next one. Attention cost grows ~ block_size^2.
    block_size: int = 64

    # vocab_size is NOT set here: it is discovered from the data (the number of
    # distinct characters). data.py fills it in and we copy it onto the config.
    vocab_size: int = -1  # placeholder, set at runtime from the dataset

    # ---- training -----------------------------------------------------------
    batch_size: int = 32          # how many sequences per gradient step
    learning_rate: float = 3e-3   # Adam step size
    beta1: float = 0.9            # Adam: decay for the 1st-moment (mean of grads)
    beta2: float = 0.99           # Adam: decay for the 2nd-moment (mean of grad^2)
    eps: float = 1e-8             # Adam: numerical fuzz to avoid divide-by-zero
    weight_decay: float = 0.0     # optional L2-ish pull toward 0 (0 = off)
    max_steps: int = 1000         # total optimizer steps (this tiny corpus converges early)
    eval_every: int = 200         # print loss / sample text this often
    seed: int = 1337              # RNG seed so runs are reproducible

    # ---- initialization -----------------------------------------------------
    # Standard deviation for the small random normal used to initialize weights.
    # Small values keep early activations well-scaled so training is stable.
    init_std: float = 0.02

    def head_dim(self) -> int:
        """Size of each attention head = d_model split across the heads."""
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        return self.d_model // self.n_heads
