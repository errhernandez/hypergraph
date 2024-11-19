
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from convolutions import HedgeConvolution, NodeConvolution
from hypergraph import HyperGraph
from hypergraph_layer import HyperGraphLayer

r"""
   This class combines an arbitrary number of HyperGraphLayer modules
   to result in a full HyperGraphConvolution module. 
"""

class HyperGraphConvolution(eqx.Module):

    conv_layers: list
    node_layers: list
    hedge_layers: list
    n_convolution_layers: int
    n_node_layers: int
    n_hedge_layers: int

    def __init__(self, 
          key: jax.Array,
          conv_layers: list[dict],
          node_layers: list[dict],
          hedge_layers: list[dict]
        ) -> None:

        r"""
        Args: 
 
          :key jax.Array: parameter initialisation random seed
 
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
               coming from the hedges. 
               Attention: The last layer of the hedge MLP should output a
               single number!
               
        """

        # set random seeds for each  layer in the convolution

        conv_key, node_key, hedge_key = jax.random.split(key, 3)

        n_convolution = len(conv_layers)
        n_node_MLP = len(node_layers)
        n_hedge_MLP = len(hedge_layers)

        convolution_keys = jax.random.split(conv_key, n_convolution)
        node_MLP_keys = jax.random.split(node_key, n_node_MLP)
        hedge_MLP_keys = jax.random.split(hedge_key, n_hedge_MLP)

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
                  key = convolution_keys[n],
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

        self.node_layers = []

        for n, layer in enumerate(node_layers):

            if 'n_node_out' in layer.keys():
                n_node_out = layer['n_node_out']
            else:
                n_node_out = n_node_in

            self.node_layers.append(
                          eqx.nn.Linear(in_features = n_node_in,
                                     out_features = n_node_out,
                                     key = node_MLP_keys[n])
                                   )

        self.n_node_layers = len(self.node_layers)

        self.hedge_layers = []

        for n, layer in enumerate(hedge_layers):

            if 'n_hedge_out' in layer.keys():
                n_hedge_out = layer['n_hedge_out']
            else:
                n_hedge_out = n_hedge_in

            self.hedge_layers.append(
                          eqx.nn.Linear(in_features = n_hedge_in,
                                     out_features = n_hedge_out,
                                     key = hedge_MLP_keys[n])
                                    )

        self.n_hedge_layers = len(self.hedge_layers)

    def __call__(self,
          hgraph: HyperGraph
        ) -> tuple[jnp.array, jnp.array]:

        node_features = hgraph.node_features
        hedge_features = hgraph.hedge_features

        # first do the hypergraph convolution

        for layer in self.conv_layers:

            node_features, hedge_features = \
             layer(node_features, hedge_features, \
             hgraph.indices())

        # then act with the MLPs

        for n, layer in enumerate(self.node_layers):

            # node_features = jax.jit(jax.vmap(layer))(node_features)
            node_features = jax.vmap(layer)(node_features)

            if n < self.n_node_layers:

               # node_features = jax.jit(jnp.tanh)(node_features)
               node_features = jnp.tanh(node_features)

        node_energy = jax.ops.segment_sum(
                          node_features,
                          segment_ids = hgraph.batch_node_index,
                          indices_are_sorted = True)

        for n, layer in enumerate(self.hedge_layers):

            # hedge_features = jax.jit(jax.vmap(layer))(hedge_features)
            hedge_features = jax.vmap(layer)(hedge_features)
           
            if n < self.n_hedge_layers:

               # hedge_features = jax.jit(jnp.tanh)(hedge_features)
               hedge_features = jnp.tanh(hedge_features)

        hedge_energy = jax.ops.segment_sum(
                           hedge_features,
                           segment_ids = hgraph.batch_hedge_index,
                           indices_are_sorted = True)

        total_energy = node_energy + hedge_energy

        return total_energy

