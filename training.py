
from typing import Callable, Tuple

import equinox as eqx
import jax
from jaxtyping import PyTree
import optax
from torch.utils.tensorboard import SummaryWriter

from hypergraph import HyperGraph
from hypergraph_dataloader import HyperGraphDataLoader
from hypergraph_model import HyperGraphConvolution

# in principle we should do what is indicated here....
# Wrap eveything -- computing gradients, running the optimiser, updating
# the model -- into a single JIT region to ensure things run as fast as possible
# @eqx.filter_jit
# but we cant, because our model contains parts that are not jittable; in 
# particular there are segment sums in the node and hedge convolutions and 
# also in the gathering of the node and hedge energies
def train_step(
        model: HyperGraphConvolution,
        loss_fn: Callable,
        opt_state: PyTree,
        optimiser: optax.GradientTransformation,
        batch: HyperGraph
    ) -> Tuple[HyperGraphConvolution, PyTree, float]:

    """Implement a single step of training"""

    # grad_fn = eqx.filter_value_and_grad(loss_fn) 

    # loss, grads = grad_fn(model, batch)

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model, batch)

    updates, opt_state = jax.jit(optimiser.update)(
        grads, opt_state, eqx.filter(model, eqx.is_array)
    )
    # updates, opt_state = optimiser.update(
    #     grads, opt_state, eqx.filter(model, eqx.is_array)
    #)
    
    model = jax.jit(eqx.apply_updates)(model, updates)

    return model, opt_state, loss

def eval_step(
        model: HyperGraphConvolution,
        loss_fn: Callable,
        batch: HyperGraph
    ) -> float:

    loss = loss_fn(model, batch)

    return loss

"""Centralises the model parameter optimisation process. """

def train_model(
      n_epochs: int,
      model: eqx.Module,
      loss_func: Callable,
      optimiser: optax.GradientTransformation,
      train_dl: HyperGraphDataLoader,
      valid_dl: HyperGraphDataLoader,
      n_epoch_0: int = 0,
      n_print: int = 1,
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
                model, loss_func, opt_state, optimiser, batch
           )
           train_running_loss += loss

       for batch in valid_dl:
           loss = eval_step(model, loss_func, batch)
           validation_running_loss += loss
       
       if epoch % n_print == 0:
    
          txt = f'{n_epoch}/{epoch}   train-loss: {train_running_loss}'
          txt += f'   validation-loss: {validation_running_loss}' 
          print(txt)

       if writer is not None:
          writer.add_scalar("Training Loss", \
                            float(train_running_loss), n_epoch)
          writer.add_scalar("Validation Loss", \
                            float(validation_running_loss), n_epoch)
                       
       # below we will have to implement checkpointing and callbacks and all that

   return model
