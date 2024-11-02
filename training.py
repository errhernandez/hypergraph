
from typing import Callable

from flax import nnx

from hypergraph import HyperGraph
from hypergraph_model import HyperGraphConvolution

def train_step(
        model: HyperGraphConvolution,
        loss_fn: Callable,
        optimizer: nnx.Optimizer,
        metrics: nnx.MultiMetric,
        batch: HyperGraph
    ) -> float:
    """Implement a single step of training"""

    grad_fn = nnx.value_and_grad(loss_fn) # has_aux=True)

    loss, grads = grad_fn(model, batch)

    metrics.update(loss=loss)
    optimizer.update(grads)

    return loss

def eval_step(
        model: HyperGraphConvolution,
        loss_fn: Callable,
        metrics: nnx.MultiMetric,
        batch: HyperGraph
    ) -> float:

    loss = loss_fn(model, batch)

    metrics.update(loss=loss)

    return loss
