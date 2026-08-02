"""
sample.py
=========

Load the weights saved by train.py and generate text. Run:

        python sample.py                 # continue from a default prompt
        python sample.py "attention "    # continue from your own prompt

This file exists to show the INFERENCE half in isolation: no training, no
gradients -- just repeatedly ask the model "what character comes next?", sample
one, append it, and repeat (see MoETransformer.generate).

Because the dataset (and therefore the vocabulary and the config) is fully
deterministic, we can rebuild an identically-shaped model here and pour the saved
weights into it. The model must be built with the SAME Config as training, or the
parameter shapes won't match the saved file.
"""

from __future__ import annotations
import sys
import numpy as np

from config import Config
from data import CharDataset
from model import MoETransformer


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "the "

    # Rebuild the exact same setup as training.
    cfg = Config()
    dataset = CharDataset()
    cfg.vocab_size = dataset.vocab_size

    rng = np.random.default_rng(0)
    model = MoETransformer(cfg, rng)

    # Load trained parameters (in place).
    try:
        model.load("moe_weights.npz")
    except FileNotFoundError:
        print("No moe_weights.npz found. Train first with:  python train.py")
        return

    # Encode the prompt, generate, decode.
    idx = dataset.encode(prompt)[None, :]
    out = model.generate(idx, max_new_tokens=400, rng=rng, temperature=0.8, top_k=10)
    print(dataset.decode(out[0]))


if __name__ == "__main__":
    main()
