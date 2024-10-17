
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from hypergraph import HyperGraph
from hypergraph_layer import HyperGraphLayer

r"""
   This class combines an arbitrary number of HyperGraphLayer modules
   to result in a full HyperGraphConvolution module. 
"""

class HyperGraphConvolution(nnx.Module):

    def __init__(self, 
          rngs: nnx.Rngs,
          layers: list[dict]
        ) -> None:

        r"""
        Args: 
 
          :rngs nnx.Rngs: parameter initialisation random seed
 
          :layers list[dict]: a list of dictionaries, one dict for each
               HyperGraphLayer; each dict must have two mandatory keys, 
               n_node_in, n_hedge_in, with int values giving the input
               node and hedge feature dimensions, respectively. Optionally, 
               additional keys n_node_out and n_hedge_out can be given; if
               not provided, these are assumed to be equal to the input values.
 
        """

        self.layers = []
 
        for n, layer in enumerate(layers):
 
            n_node_in = layer['n_node_in']
            n_hedge_in = layer['n_hedge_in']
 
            if 'n_node_out' in layer.keys():
               n_node_out = layer['n_node_out']
            else:
               n_node_out = n_node_in
 
            if 'n_hedge_out' in layer.keys():
               n_hedge_out = layer['n_hedge_out']
            else:
               n_hedge_out = n_hedge_in
 
            self.layers.append( HyperGraphLayer(
                  rngs = rngs,
                  n_node_in = n_node_in, 
                  n_hedge_in = n_hedge_in,
                  n_node_out = n_node_out,
                  n_hedge_out = n_hedge_out
                  ) )
 
        self.n_layers = len(layers)

    def __call__(self,
          hgraph: HyperGraph
        ) -> tuple[jnp.array, jnp.array]:

        node_features = hgraph.node_features
        hedge_features = hgraph.hedge_features

        for layer in self.layers:

            node_features, hedge_features = \
             layer(node_features, hedge_features, \
             hgraph.data_dict())

        return node_features, hedge_features

