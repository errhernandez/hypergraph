
import equinox as eqx
import jax
import jax.numpy as jnp

from hypergraph import HyperGraph
from hypergraph_model import HyperGraphConvolution

# this function should calculate the forces with respect to atomic positions

def evaluate_energy(
        params, 
        static, 
        node_features: jnp.ndarray,
        hedge_features: jnp.ndarray,
        hypergraph: HyperGraph
    ) -> float:

    model = eqx.combine(params, static)

    energy = model(node_features, hedge_features, hypergraph)

    return energy[0,0]
