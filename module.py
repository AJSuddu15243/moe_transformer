"""
module.py
=========

THE BUILDING BLOCK OF EVERYTHING ELSE.

Every layer in this project (Linear, LayerNorm, Attention, MoE, ...) inherits
from the tiny `Module` base class defined here. The whole point of this file is
to answer two questions in ONE consistent way, so we never have to think about
them again:

    1. "Where do a layer's learnable numbers live?"        -> self.params
    2. "Where do the gradients for those numbers live?"    -> self.grads

and to give us a way to walk an entire tree of nested layers and collect every
(name, parameter, gradient) triple. The optimizer uses that walk to update the
whole network with a single loop.

------------------------------------------------------------------------------
WHY BUILD THIS AT ALL? (the mental model)
------------------------------------------------------------------------------
A neural network is a big mathematical function

        loss = f(inputs; theta)

where `theta` is the giant bag of all learnable numbers (weights, biases,
embedding tables, LayerNorm scales, ...). "Training" means:

    - run the function forward to get `loss`                (the forward pass)
    - compute d loss / d theta  for EVERY number in theta   (the backward pass)
    - nudge theta a little bit in the direction that lowers loss (the optimizer)

Frameworks like PyTorch build a graph and do the backward pass automatically.
We are doing it BY HAND so you can see the calculus. To stay organized we adopt
one rigid convention that every layer obeys:

    layer.forward(x)      returns the output, and stashes whatever intermediate
                          values the backward pass will need (in self.cache or
                          plain attributes).

    layer.backward(dout)  receives `dout` = d loss / d (this layer's output),
                          fills in self.grads[...] = d loss / d (each param),
                          and RETURNS d loss / d (this layer's input) so the
                          previous layer can continue the chain.

That RETURN value is the chain rule in action: gradients flow backwards through
the network, each layer handing the next one upstream its share.
------------------------------------------------------------------------------
"""

from __future__ import annotations
from typing import Dict, Iterator, Tuple
import numpy as np


class Module:
    """Base class for every layer.

    A Module owns:
      - self.params:   dict[str, np.ndarray]  the learnable arrays
      - self.grads:    dict[str, np.ndarray]  d loss / d param, same shapes
      - self._children: dict[str, Module]     sub-modules (for nesting)

    Containers (like a transformer Block, which holds an attention layer and an
    MoE layer) register their sub-layers in self._children so that param_items()
    can recurse and find *every* parameter in the network.
    """

    def __init__(self) -> None:
        # The learnable numbers. Example for a Linear layer: {"W": (in,out), "b": (out,)}.
        self.params: Dict[str, np.ndarray] = {}
        # Gradient for each param, filled in during backward(). Same keys, same shapes.
        self.grads: Dict[str, np.ndarray] = {}
        # Sub-modules, keyed by a short name. Used only by container layers.
        self._children: Dict[str, "Module"] = {}

    # -- registration ---------------------------------------------------------
    def register(self, name: str, child: "Module") -> "Module":
        """Record a sub-module so param_items() can reach its parameters.

        We return `child` so callers can write:  self.attn = self.register("attn", Attention(...))
        which both stores the object as an attribute (for use in forward/backward)
        AND registers it for parameter collection. Two birds, one line.
        """
        self._children[name] = child
        return child

    # -- parameter collection -------------------------------------------------
    def param_items(self, prefix: str = "") -> Iterator[Tuple[str, np.ndarray, np.ndarray]]:
        """Yield (full_name, param_array, grad_array) for THIS module and all
        descendants, depth-first.

        `full_name` is a dotted path like "block0.attn.Wq.W" that is unique
        across the whole network. The optimizer uses it as a dictionary key to
        remember per-parameter state (Adam's moving averages) between steps.

        IMPORTANT: we yield the *live* array objects, not copies.
          - The param array is mutated in place by the optimizer (theta -= step),
            so the model immediately sees the updated weights.
          - The grad array is whatever self.grads[k] points to at call time.
            Because we re-run param_items() every optimization step, we always
            read the freshly computed gradients from the latest backward pass.
        """
        for k, p in self.params.items():
            yield prefix + k, p, self.grads[k]
        for name, child in self._children.items():
            # Recurse. The dotted prefix keeps names globally unique.
            yield from child.param_items(prefix + name + ".")

    def num_params(self) -> int:
        """Total count of scalar learnable numbers in this module (and children).
        Handy for the 'keep it small' sanity check printed at startup."""
        return int(sum(p.size for _, p, _ in self.param_items()))
