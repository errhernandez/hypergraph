
from functools import partial

import equinox as eqx
import jax.numpy as jnp
import jax
from optax.losses import l2_loss

from hypergraph import HyperGraph

# @partial(jax.jit, static_argnames=['num_segments'])
# @eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def loss_function(
         model: eqx.Module,
         node_features: jnp.array,
         hedge_features: jnp.array,
         indices: dict,
         target_key: str = 'U0') -> float:

    prediction = model(node_features, hedge_features, indices)

    ground_truth = jnp.array(indices['targets'][target_key])

    # indexing of [:-1] is to exclude influence of the last 
    # empty hypergraph in the batch, which should not be taken
    # into account

    loss = l2_loss(prediction[:-1], ground_truth[:-1]).sum()

    return loss
