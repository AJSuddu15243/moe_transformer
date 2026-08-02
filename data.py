"""
data.py
=======

The world's simplest tokenizer + data loader: CHARACTER-LEVEL language modeling.

WHAT IS THE TASK?
-----------------
We give the model a string of characters and ask it to predict the NEXT
character at every position. If the text is "hello", then from the prefix
"hell" it should predict "o". Do this well over a whole corpus and the model
learns spelling, word boundaries, and a bit of grammar -- all from raw bytes,
no hand-built features.

Character-level means our "tokens" are just individual characters. The
"vocabulary" is the set of distinct characters in the training text. This keeps
everything tiny and transparent: you can literally read the token ids.

HOW TRAINING DATA IS SHAPED
---------------------------
The model sees fixed-length windows of `block_size` characters. For a window of
input characters x[0..T-1], the target is the SAME window shifted left by one:
y[t] = x[t+1]. So at every position t the model predicts the character that
actually follows. One window therefore gives us T supervised (input -> next)
pairs at once, which is why transformers train so efficiently.

    text:    "to be or not"
    x (in):   t o   b e   o r   n o   ...   (a length-T slice)
    y (tgt):  o   b e   o r   n o   t   ...   (the same slice shifted by 1)
"""

from __future__ import annotations
from typing import Tuple
import numpy as np


# A small, self-contained training corpus so the project needs no downloads.
# It's a short passage repeated/varied enough to give the model patterns to
# learn (common words, punctuation, capitalization). Feel free to replace this
# with any text you like -- the vocabulary adapts automatically.
TEXT = (
    "the mixture of experts routes each token to a few specialized networks.\n"
    "attention lets every position look back at the ones before it.\n"
    "a small model can still learn the shape of language from characters.\n"
    "gradients flow backward through every layer to teach the weights.\n"
    "softmax turns raw scores into a probability over the next character.\n"
    "the router decides which experts should think about this token.\n"
    "layer norm keeps the numbers well behaved as they move through depth.\n"
    "we train by lowering the loss one small adam step at a time.\n"
) * 12  # repeat so there is enough text to sample many windows from


class CharDataset:
    """Turns a raw string into integer token ids and serves random batches."""

    def __init__(self, text: str = TEXT):
        # ---- build the vocabulary ------------------------------------------
        # sorted() makes the mapping deterministic across runs.
        chars = sorted(set(text))
        self.vocab_size = len(chars)

        # stoi = "string to integer": maps each character to its id.
        # itos = "integer to string": the inverse, used when we generate text.
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

        # Encode the entire corpus once into a 1-D array of token ids.
        # dtype int64 because these are indices into embedding tables.
        self.data = np.array([self.stoi[c] for c in text], dtype=np.int64)

    # -- encode / decode helpers ---------------------------------------------
    def encode(self, s: str) -> np.ndarray:
        """string -> array of token ids."""
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids) -> str:
        """array/list of token ids -> string."""
        return "".join(self.itos[int(i)] for i in ids)

    # -- batching -------------------------------------------------------------
    def get_batch(self, batch_size: int, block_size: int, rng: np.random.Generator
                  ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample a random batch of (inputs, targets).

        Returns:
            x: int64 array (batch_size, block_size)  -- input token ids
            y: int64 array (batch_size, block_size)  -- next-token targets

        We pick `batch_size` random start positions. From each start we take a
        window of length block_size for x, and the window shifted by +1 for y.
        Because y is x shifted by one, target y[:, t] is the character that
        follows input x[:, t] -- exactly the next-token prediction objective.
        """
        # Highest valid start index: we need block_size+1 characters available
        # (block_size for x, plus one more so y can shift by one).
        max_start = len(self.data) - block_size - 1

        # Random start offsets, one per sequence in the batch.
        starts = rng.integers(0, max_start, size=batch_size)

        # Gather the windows. Building with a list comprehension keeps it
        # obvious; for tiny batches the speed cost is irrelevant.
        x = np.stack([self.data[s : s + block_size] for s in starts])
        y = np.stack([self.data[s + 1 : s + 1 + block_size] for s in starts])
        return x, y
