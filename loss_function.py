
from flax import nnx
import jax.numpy as jnp
from optax.losses import l2_loss

from hypergraph import HyperGraph

def loss_function(
         model: nnx.Module,
         hgraph: HyperGraph,
         target_key: str = 'U0') -> float:

    prediction = model(hgraph)

    ground_truth = jnp.array(hgraph.targets[target_key])

    loss = l2_loss(prediction, ground_truth).sum()

    return loss
