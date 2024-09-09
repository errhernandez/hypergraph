
from typing import Optional

import jax
import jax.numpy as jnp
import equinox as eqx

from hypergraph import HyperGraph

# first we are going to define two sub-modules, one node convolution, one for hedge convolution
# these will then become part of the HyperGraphModule model

class NodeConvolution(eqx.Module):

    def __init__(self,
            key: jnp.ndarray,
            n_node_features_in: int,
            n_hedge_features: int,
            n_node_features_out: Optional[int] = None
            ) -> None:

        if n_node_features_out is None: n_node_features_out = n_node_features_in

        # define two linear layers (eventually these may become full MLPs)
        # one for transforming node features, and one for 

        key1, key2 = jax.random.split(key)

        self.node_message = \
            eqx.nn.Linear(n_node_features_in, n_node_features_out, key1)

        self.hedge_scaling = \
            eqx.nn.Linear(n_hedge_features, n_node_features_out, key2)

    def __call__(self,
            node_input,
            node_senders,
            node_receivers, 
            node_adjacency,

        # first convolve node features

        sender_features = node_input[node_senders]

        messages = self.node_message(sender_features)

        scaled_messages = node_adjacency * messages

        gathered_messages = jax.ops.segment_sum(scaled_messages, receivers)

        # now use hedge embeddings to co-embed node embeddings

class HyperGraphModule(eqx.Module):

    def __init__(self, n_node_in, n_hedge_in, n_node_out, n_hedge_out, key) -> None:
        r"""

        Args:
           n_node_in (int): input node features size
           n_hedge_in (int): input hedge features size
           n_node_out (int): output node embeddings size
           n_hedge_out (int): output hedge embeddings size
           key (jax random key: initialization key

        """

        self.linearnn


    def __call__(self, hgraph: HyperGraph)

