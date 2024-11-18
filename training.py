
from typing import Callable

import equinox as eqx
# from flax import nnx
import jax
import optax
from torch.utils.tensorboard import SummaryWriter

from hypergraph import HyperGraph
from hypergraph_dataloader import HyperGraphDataLoader
from hypergraph_model import HyperGraphConvolution

def train_step(
        model: HyperGraphConvolution,
        loss_fn: Callable,
        optimizer: optax.GradientTransformation,
        batch: HyperGraph
    ) -> float:
    """Implement a single step of training"""

    grad_fn = eqx.filter_value_and_grad(loss_fn) # has_aux=True)

    loss, grads = grad_fn(model, batch)

    optimizer.update(grads)

    return loss

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
      optimizer: optax.GradientTransformation, 
      train_dl: HyperGraphDataLoader,
      valid_dl: HyperGraphDataLoader,
      n_epoch_0: int = 0,
      n_print: int = 1,
      writer: SummaryWriter = None
		) -> None:

   print("epoch-run/epoch      train-loss      validation-loss")
   print("----------------------------------------------------")
 
   for epoch in range(n_epochs):

       n_epoch = n_epoch_0 + epoch

       train_running_loss = 0.0
       validation_running_loss = 0.0

       for batch in train_dl:
           loss = train_step(model, loss_func, optimizer, batch)
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
