
import equinox as eqx
import jax
import jax.numpy as jnp

from hypergraph import HyperGraph
from hypergraph_model import HyperGraphConvolution

# this function should calculate the forces with respect to atomic positions

def evaluate_energy(
        model: HyperGraphConvolution,
        

def calculate_forces(
        model: HyperGraphConvolution,
        sample: HyperGraph
    ) -> jnp.ndarray:

    
