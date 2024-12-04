
from typing import Callable, Tuple
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree
import optax
from torch.utils.tensorboard import SummaryWriter

from checkpointing import checkpoint_save
from hypergraph import HyperGraph
from hypergraph_dataloader import HyperGraphDataLoader
from hypergraph_model import HyperGraphConvolution

# in principle we should do what is indicated here....
# Wrap eveything -- computing gradients, running the optimiser, updating
# the model -- into a single JIT region to ensure things run as fast as possible
# but we cant, because our model contains parts that are not jittable; in 
# particular there are segment sums in the node and hedge convolutions and 
# also in the gathering of the node and hedge energies

# @partial(jax.jit, static_argnames=['num_segments'])
# @eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def train_step(
        model: HyperGraphConvolution,
        loss_fn: Callable,
        opt_state: PyTree,
        optimiser: optax.GradientTransformation,
        node_features: jnp.array,
        hedge_features: jnp.array,
        indices: dict
    ) -> Tuple[HyperGraphConvolution, PyTree, float]:

    """Implement a single step of training"""

    # grad_fn = eqx.filter_value_and_grad(loss_fn) 

    # loss, grads = grad_fn(model, batch)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model,
                       node_features,
                       hedge_features,
                       indices
                  )

    updates, opt_state = jax.jit(optimiser.update)(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    # updates, opt_state = optimiser.update(
    #     grads, opt_state, eqx.filter(model, eqx.is_array)
    #)
    
    # model = jax.jit(eqx.apply_updates)(model, updates)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss

# @eqx.filter_jit
# @eqx.debug.assert_max_traces(max_traces=1)
def eval_step(
        model: HyperGraphConvolution,
        loss_fn: Callable,
        node_features: jnp.array,
        hedge_features: jnp.array,
        indices: dict
    ) -> float:

    loss = loss_fn(model,
                   batch.node_features,
                   batch.hedge_features,
                   batch.indices())

    return loss

"""Centralises the model parameter optimisation process. """

def train_model(
      n_epochs: int,
      model: eqx.Module,
      hyperparameters: dict,
      loss_func: Callable,
      optimiser: optax.GradientTransformation,
      train_dl: HyperGraphDataLoader,
      valid_dl: HyperGraphDataLoader,
      n_epoch_0: int,
      n_print: int,
      checkpoint_file: str,
      n_checkpoint_freq: int, 
      writer: SummaryWriter = None
		) -> eqx.Module:

   # filter the model to separate arrays from everything else

   opt_state = optimiser.init(eqx.filter(model, eqx.is_array))

   print("epoch-run/epoch      train-loss      validation-loss")
   print("----------------------------------------------------")
 
   for epoch in range(n_epochs):

       n_epoch = n_epoch_0 + epoch

       train_running_loss = 0.0
       validation_running_loss = 0.0

       for batch in train_dl:
           model, opt_state, loss = train_step(
                                                model,
                                                loss_func,
                                                opt_state,
                                                optimiser,
                                                batch.node_features,
                                                batch.hedge_features,
                                                batch.indices()
                                              )

           train_running_loss += loss

       for batch in valid_dl:
           loss = eval_step(
                             model,
                             loss_func,
                             batch.node_features,
                             batch.hedge_features,
                             batch.indices()
                           )

           validation_running_loss += loss

       if epoch % n_print == 0:
    
          txt = f'{n_epoch}/{epoch}   train-loss: {train_running_loss}'
          txt += f'   validation-loss: {validation_running_loss}' 
          print(txt)

       if epoch % n_checkpoint_freq == 0:
          checkpoint_save(checkpoint_file, hyperparameters, model)

       if writer is not None:
          writer.add_scalar("Training Loss", \
                            float(train_running_loss), n_epoch)
          writer.add_scalar("Validation Loss", \
                            float(validation_running_loss), n_epoch)
                       
       # below we will have to implement checkpointing and callbacks and all that

   return model
