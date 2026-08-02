"""
train.py
========

The training loop: the place where forward, loss, backward, and the optimizer
come together to actually teach the model. Run it with:

        python train.py

WHAT ONE TRAINING STEP DOES
---------------------------
    1. Sample a random batch of (inputs x, targets y) from the corpus.
    2. FORWARD:  logits = model(x)         -- predictions at every position.
    3. LOSS:     ce = cross_entropy(logits, y);  total = ce + aux_load_balance.
    4. BACKWARD: model.backward(dlogits)   -- fill every parameter's gradient.
    5. UPDATE:   optimizer.step()          -- Adam nudges every weight downhill.
Repeat thousands of times; watch the loss fall and the samples get more
word-like. Periodically we print the loss and generate a snippet so you can SEE
the model learning to spell.

WHAT "LOSS" MEANS HERE
----------------------
The loss is the average negative log-probability the model assigns to the true
next character (in nats). At random init it is about ln(vocab_size). As the model
learns character statistics and then words, it drops well below that. There is no
"accuracy" for a generative char model -- lower loss = better next-char guesses.
"""

from __future__ import annotations
import numpy as np

from config import Config
from data import CharDataset
from model import MoETransformer
from optimizer import Adam
from losses import cross_entropy


def main():
    cfg = Config()
    rng = np.random.default_rng(cfg.seed)

    # ---- data ----
    dataset = CharDataset()
    cfg.vocab_size = dataset.vocab_size          # discovered from the text
    print(f"corpus chars: {len(dataset.data)}   vocab size: {cfg.vocab_size}")
    print(f"random-guess loss (ln V) = {np.log(cfg.vocab_size):.3f}")

    # ---- model + optimizer ----
    model = MoETransformer(cfg, rng)
    print(f"model parameters: {model.num_params():,}")
    opt = Adam(model, lr=cfg.learning_rate, beta1=cfg.beta1, beta2=cfg.beta2,
               eps=cfg.eps, weight_decay=cfg.weight_decay)

    # ---- training loop ----
    for step in range(1, cfg.max_steps + 1):
        # 1. batch
        x, y = dataset.get_batch(cfg.batch_size, cfg.block_size, rng)

        # 2. forward
        logits = model.forward(x)

        # 3. loss (cross-entropy + MoE load-balancing auxiliary)
        ce, dlogits = cross_entropy(logits, y)
        total = ce + model.aux_loss

        # 4. backward: dlogits seeds the chain; the aux-loss gradient is added
        #    internally inside each MoE layer's backward.
        model.backward(dlogits)

        # 5. optimizer update
        opt.step()

        # ---- logging / sampling ----
        if step % cfg.eval_every == 0 or step == 1:
            sample = generate_sample(model, dataset, rng, n=160)
            print(f"\nstep {step:>5} | loss {ce:6.3f} | aux {model.aux_loss:6.4f} "
                  f"| total {total:6.3f}")
            print("  sample: " + sample.replace("\n", "\n          "))

    # ---- save the trained weights ----
    model.save("moe_weights.npz")
    print("\nsaved weights to moe_weights.npz  (use `python sample.py` to generate more)")


def generate_sample(model, dataset, rng, n=160, prompt="the "):
    """Generate `n` characters continuing from `prompt` and return the string."""
    idx = dataset.encode(prompt)[None, :]                       # (1, len(prompt))
    out = model.generate(idx, max_new_tokens=n, rng=rng, temperature=0.8, top_k=10)
    return dataset.decode(out[0])


if __name__ == "__main__":
    main()
