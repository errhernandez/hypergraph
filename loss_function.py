
from flax import nnx
import jax.numpy as jnp
from optax.losses import l2_loss

# from hypergraph import HyperGraph

def loss_function(
         model: nnx.Module,
         hgraph_data: dict[str, jnp.ndarray],
         target_key: str = 'U0') -> float:

    prediction = model(hgraph_data)

    ground_truth = jnp.array(hgraph_data['targets'][target_key])

    loss = l2_loss(prediction, ground_truth).sum()

    return loss
