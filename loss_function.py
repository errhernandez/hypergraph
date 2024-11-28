
import equinox as eqx
import jax.numpy as jnp
from optax.losses import l2_loss

from hypergraph import HyperGraph

def loss_function(
         model: eqx.Module,
         hgraph: HyperGraph,
         target_key: str = 'U0') -> float:

    prediction = model(hgraph)

    ground_truth = jnp.array(hgraph.targets[target_key])

    # indexing of [:-1] is to exclude influence of the last 
    # empty hypergraph in the batch, which should not be taken
    # into account

    loss = l2_loss(prediction[:-1], ground_truth[:-1]).sum()

    return loss
