
import json
import os
import sys

import equinox as eqx
import jax
import jax.numpy as jnp

from hypergraph_model import builder

def checkpoint_save(filename, hyperparams, model) -> None:

    # I suspect that json is appending, so remove file
    # to make sure that we save the last params only
    if os.path.exists(filename):
       os.remove(filename)

    with open(filename, 'wb') as f:
        hyperparam_str = json.dumps(hyperparams)
        f.write((hyperparam_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)

def checkpoint_load(filename) -> eqx.Module:

    with open(filename, 'rb') as f:
        hyperparams = json.loads(f.readline().decode())
        model = builder(key=jax.random.PRNGKey(0), **hyperparams)
        return eqx.tree_deserialise_leaves(f, model)

