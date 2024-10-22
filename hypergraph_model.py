
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
          conv_layers: list[dict],
          node_layers: list[dict],
          hedge_layers: list[dict]
        ) -> None:

        r"""
        Args: 
 
          :rngs nnx.Rngs: parameter initialisation random seed
 
          :conv_layers list[dict]: a list of dictionaries, one dict for each
               HyperGraphLayer; each dict must have two mandatory keys, 
               n_node_in, n_hedge_in, with int values giving the input
               node and hedge feature dimensions, respectively. Optionally, 
               additional keys n_node_out and n_hedge_out can be given; if
               not provided, these are assumed to be equal to the input values.

          :node_layers list[dict]: a list of dictionaries defining the layers
               of a Multilayer Perceptron (MLP) responsible for calculating 
               the energy contribution of the nodes. Attention: the last layer
               of the node MLP should output a single number!

          :hedge_layers list[dict]: similar to node_layers, but for hedges, 
               a dict defining an MLP for calculating the energy contribution
               coming from the hedges. Here it must be taken into account that
               hedges (electrons) may have properties (hedge_properties) that
               are not subject to the convolution (e.g. spin) but that nevertheless
               should be taken into account to get their energy contribution.
               Attention: The last layer of the hedge MLP should output a
               single number!
               
        """

        # first set-up the hyper-graph convolution layers

        self.conv_layers = []
 
        for n, layer in enumerate(conv_layers):
 
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
 
            self.conv_layers.append( HyperGraphLayer(
                  rngs = rngs,
                  n_node_in = n_node_in, 
                  n_hedge_in = n_hedge_in,
                  n_node_out = n_node_out,
                  n_hedge_out = n_hedge_out
                  ) )
 
        self.n_convolution_layers = len(self.conv_layers)

        # next set up the MLPs for nodes and hedges
        # to calculate their respective contributions to
        # the total energy; their input size must accord
        # with the convolution output for nodes and hedges
        # respectively

        last_layer = conv_layers[-1]

        if 'n_node_out' in last_layer.keys():
           n_node_in = last_layer['n_node_out']
        else:
           n_node_in = last_layer['n_node_in']

        if 'n_hedge_out' in last_layer.keys():
           n_hedge_in = last_layer['n_hedge_out']
        else:
           n_hedge_in = last_layer['n_hedge_in']

        self.n_node_layers = len(node_layers)
        self.node_layers = []

        for n, layer in enumerate(node_layers):

            if 'n_node_out' in layer.keys():
                n_node_out = layer['n_node_out']
            else:
                n_node_out = n_node_in

            self.node_layers.append(
                 nnx.vmap(
                          nnx.Linear(in_features = n_node_in,
                                     out_features = n_node_out,
                                     rngs = rngs),
                          in_axes = 0,
                          out_axes = 0
                         )
                                   )

        self.n_hedge_layers = len(hedge_layers)
        self.hedge_layers = []

        for n, layer in enumerate(hedge_layers):

            if 'n_hedge_out' in layer.keys():
                n_hedge_out = layer['n_hedge_out']
            else:
                n_hedge_out = n_hedge_in

            self.hedge_layers.append(
                 nnx.vmap(
                          nnx.Linear(in_features = n_hedge_in,
                                     out_features = n_hedge_out,
                                     rngs = rngs),
                          in_axes = 0,
                          out_axes = 0
                         )
                                    )


    def __call__(self,
          hgraph: HyperGraph
        ) -> tuple[jnp.array, jnp.array]:

        node_features = hgraph.node_features
        hedge_features = hgraph.hedge_features

        # first do the hypergraph convolution

        for layer in self.conv_layers:

            node_features, hedge_features = \
             layer(node_features, hedge_features, \
             hgraph.data_dict())

        # then act with the MLPs

        for n, layer in enumerate(self.node_layers):

            node_features = layer(node_features)

            if n < self.n_node_layers:

               node_features = jnp.tanh(node_features)

        node_energy = jnp.sum(node_features, axis=0)

        for n, layer in enumerate(self.hedge_layers):

            hedge_features = layer(hedge_features)
           
            if n < self.n_hedge_layers:

               hedge_features = jnp.tanh(hedge_features)

        hedge_energy = jnp.sum(hedge_features, axis=0)

        total_energy = node_energy + hedge_energy

        return total_energy

