# A Mixture-of-Experts Transformer, from scratch in pure NumPy

A complete, tiny (~175k parameter) **Mixture-of-Experts (MoE) transformer** —
forward pass **and** hand-written backward pass — trained as a character-level
language model. No PyTorch, no autograd, no `import` beyond NumPy. Every file is
heavily commented with the math *and* the intuition so you can learn how a modern
sparse transformer actually works, one derivative at a time.

The whole point: **you can read every number.** Nothing is hidden behind a
framework. The gradients are derived by hand and then *proven correct* against
finite differences (`gradcheck.py`).

---

## Quick start

```bash
cd moe_transformer

python gradcheck.py     # prove every hand-derived gradient matches finite differences
python train.py         # train the char-LM (~90s on a laptop CPU); saves moe_weights.npz
python sample.py "the " # generate text from the trained model
```

You'll watch the loss fall from ~3.30 (random guessing) to ~0.1 and the samples
go from `ttety srsadieoia` to real sentences.

---

## What is a Mixture of Experts, in one picture

A normal transformer block is **attention → one feed-forward network (FFN)**.
MoE replaces that single FFN with **many expert FFNs + a router** that, *per
token*, sends each token to only a few experts:

```
                    a token's vector x
                           |
                        [ router ]  -> a score for each expert
                           |
                     softmax + pick top-k          <- the "sparse" decision
                     /       \
              gate*Expert_2   gate*Expert_5         <- only these 2 run for x
                     \       /
                        (sum)  -> the token's new vector
```

The trick: you get the **capacity** of many experts but only pay the **compute**
of `top_k` of them per token. Different experts specialize on different inputs.
A learned "router" makes the choice, and a small **load-balancing loss** stops
the router from lazily dumping everything on one expert.

---

## The architecture, end to end

```
token ids (B, T)
   │
   ▼  Embedding: tok[id] + pos[t]                     (model.py: Embedding)
   ▼
 ┌─────────────────────────── Block ×N ──────────────────────────┐   (block.py)
 │   x = x + Attention( LayerNorm(x) )   ← tokens communicate      │
 │   x = x + MoE(       LayerNorm(x) )   ← per-token computation   │
 └───────────────────────────────────────────────────────────────┘
   │
   ▼  final LayerNorm                                  (layernorm.py)
   ▼  LM head: Linear(d_model → vocab)                 (linear.py)
   ▼
logits (B, T, vocab)  →  softmax cross-entropy vs. next token   (losses.py)
```

`B` = batch, `T` = sequence length, `N` = number of blocks.

---

## Suggested reading order

The files are written to be read like chapters. Follow the data, then follow the
gradients back:

1. **`module.py`** — the 30-line base class. Establishes the one convention every
   layer follows: `forward` caches, `backward` returns the input gradient.
2. **`linear.py`** — the matmul + bias layer and its `X^T @ dY` / `dY @ W^T`
   gradients. Reused *everywhere*; understand this and half the backward pass is
   done.
3. **`activations.py`** — GELU and its exact derivative (the only nonlinearity).
4. **`layernorm.py`** — normalization and the compact "subtract the mean and the
   xhat-projection" input gradient.
5. **`attention.py`** — causal multi-head self-attention. The only new gradient
   idea is the **softmax Jacobian**: `dscores = p * (dp - Σ dp·p)`.
6. **`moe.py`** — the heart of the project: router, top-k selection,
   renormalized gating, experts, the load-balancing auxiliary loss, and the
   backward pass through all of it.
7. **`block.py`** — how residual connections turn the pieces into a trainable
   deep stack (and why the gradient "highway" matters).
8. **`losses.py`** — softmax cross-entropy and its famously clean
   `probs - onehot` gradient.
9. **`model.py`** — assembles embeddings + blocks + head, plus text generation
   and weight save/load.
10. **`optimizer.py`** — Adam: per-parameter adaptive step sizes from moving
    averages of the gradient and its square.
11. **`train.py`** — the loop that ties it together.
12. **`gradcheck.py`** — the proof. Read this to convince yourself the rest is
    right.

---

## The five gradient ideas you actually need

Everything in the backward pass is one of these, applied repeatedly:

| Operation | Forward | Gradient |
|---|---|---|
| Linear | `y = xW + b` | `dW = xᵀdy`, `db = Σdy`, `dx = dy Wᵀ` |
| Softmax | `p = softmax(s)` | `ds = p * (dp − Σⱼ dpⱼ pⱼ)` |
| Cross-entropy(softmax) | `−log p[target]` | `dlogits = (p − onehot)/N` |
| LayerNorm | normalize over features | `dx = invstd/D · (D·dx̂ − Σdx̂ − x̂·Σ(dx̂·x̂))` |
| Residual | `y = x + f(x)` | gradient **splits** and flows both paths, then adds |

The MoE gating adds one more: differentiating the **renormalized top-k** weights
`gateₑ = pₑ / Σⱼ∈S pⱼ` gives, for each chosen expert `a`,
`dpₐ = (dGₐ − Σₑ∈S gateₑ·dGₑ) / denom`. It's derived line-by-line in `moe.py`.

---

## Default hyperparameters (all in `config.py`)

| | value | meaning |
|---|---|---|
| `d_model` | 64 | width of the residual stream |
| `n_heads` | 4 | attention heads |
| `n_layers` | 2 | transformer blocks |
| `n_experts` | 4 | experts per MoE layer |
| `top_k` | 2 | experts actually run per token |
| `d_ff` | 128 | hidden width inside each expert |
| `block_size` | 64 | context length |
| `aux_coeff` | 0.01 | strength of load-balancing loss |

≈ **175k parameters** total. Deliberately small so it trains on a CPU in NumPy in
about a minute and a half.

---

## One honesty note about efficiency

For clarity and easy-to-verify backprop, this code runs **every expert on every
token** and multiplies the un-chosen ones by a gate of `0`. A *production* MoE
instead **dispatches**: it gathers only the tokens each expert needs and never
computes the rest — that's where the compute savings live. The **math of routing,
gating, and backprop is identical**; only the bookkeeping differs. `moe.py` flags
exactly where the sparsity would kick in.

---

## Things to try (to cement understanding)

- Set `n_experts=1, top_k=1` → you get an ordinary (dense) transformer. Compare.
- Set `aux_coeff=0` and watch whether the router **collapses** onto a few experts
  (print `f` = per-expert token fractions inside `moe.py`).
- Increase `top_k` toward `n_experts` → routing gets denser, loss usually drops
  but compute grows.
- Replace the corpus in `data.py` with your own text; the vocabulary adapts.
- Add a third expert layer, or widen `d_model`, and watch the parameter count and
  training time move.
