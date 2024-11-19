
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from hypergraph import HyperGraph

# first we are going to define two sub-modules, one node convolution, one for hedge convolution
# these will then become part of the HyperGraphModule model

class NodeConvolution(eqx.Module):

    r"""
    This module takes care of node convolution in a hypergraph. In essence, a node receives
    messages from its neighbour nodes (including itself) that are obtained from their 
    respective feature vectors filtered through a linear layer (self.node_message) and added up. 
    The result is then Hadamard-scaled by a vector resulting from the features of all edges 
    connected to the node, after filtering them through a second linear layer (self.hedge_scaling). 
    """

    node_message: eqx.nn.Linear
    hedge_scaling: eqx.nn.Linear

    def __init__(self,
            key: jax.Array,
            n_node_features_in: int,
            n_hedge_features: int,
            n_node_features_out: Optional[int] = None,
            ) -> None:

        r"""
        Args:
           key (jax.Array): random key for initialisation purposes.
           n_node_features_in (int): input node feature vector dimension.
           n_hedge_features (int): hedge feature vector dimension.
           n_node_features_out (int, optional): output node feature vector dimension;
             if not given assumed to be same as n_node_features_in

        """

        key1, key2 = jax.random.split(key,2)

        if n_node_features_out is None: 
           n_node_features_out = n_node_features_in

        # define two linear layers (eventually these may become full MLPs)
        # one for transforming node features, and one for obtaining
        # hedge scaling  

        self.node_message = eqx.nn.Linear(
              in_features = n_node_features_in,
              out_features = n_node_features_out,
              key = key1
        )

        self.hedge_scaling = eqx.nn.Linear(
             in_features = n_hedge_features,
             out_features = n_node_features_out,
             key = key2
        )

    def __call__(self,
            node_features: jnp.ndarray,
            hedge_features: jnp.ndarray,
            hgraph_data: dict[str, jnp.ndarray]
        ) -> jnp.ndarray:

        r"""
        Args:
           node_features: array of input node features
           hedge_features: array of hedge_feature vectors
           hgraph_data: a pytree dictionary containing HyperGraph index arrays

        Shapes:
           -**inputs:**
           node_features [:, n_node_features_in], first dimension runs over nodes
           hedge_features [:, n_hedge_features], first dimension runs over hedges

           -**outputs:**
           output_node_features [:, n_node_features_out]

        """

        node_senders = hgraph_data['node_senders']
        node_receivers = hgraph_data['node_receivers']
        node_convolution = hgraph_data['node_convolution']
        
        hedge2node_senders = hgraph_data['hedge2node_senders']
        hedge2node_receivers = hgraph_data['hedge2node_receivers']
        hedge2node_convolution = hgraph_data['hedge2node_convolution']

        # first convolve node features

        sender_features = node_features[node_senders]

        # messages = jax.jit(jax.vmap(self.node_message))(sender_features)
        messages = jax.vmap(self.node_message)(sender_features)

        scaled_messages = node_convolution * messages

        gathered_messages = jax.ops.segment_sum(scaled_messages,
                               node_receivers)

        # now use hedge embeddings to co-embed node embeddings

        hedge_sender_features = hedge_features[hedge2node_senders]

        # hedge_messages = jax.jit(
        #           jax.vmap(self.hedge_scaling))(hedge_sender_features)
        hedge_messages = \
                  jax.vmap(self.hedge_scaling)(hedge_sender_features)

        scaled_hedge_messages = hedge2node_convolution * hedge_messages

        gathered_scaling = jax.ops.segment_sum(scaled_hedge_messages,
                              hedge2node_receivers)

        # output = jax.jit(jnp.tanh)(gathered_scaling * gathered_messages)
        output = jnp.tanh(gathered_scaling * gathered_messages)

        return output

class HedgeConvolution(eqx.Module):

    r"""
    This module takes care of hedge convolution in a hypergraph. It is a mirror image of
    NodeConvolution above, but for hedges instead of nodes. In essence, an hedge receives
    messages from its neighbouring hedges (including itself) that are obtained from their 
    respective feature vectors filtered through a linear layer (self.hedge_message) and added up. 
    The result is then Hadamard-scaled by a vector resulting from the features of all nodes 
    connected to the hedge, after filtering them through a second linear layer (self.node_scaling). 
    """

    hedge_message: eqx.nn.Linear
    node_scaling: eqx.nn.Linear

    def __init__(self,
            key: jax.Array,
            n_hedge_features_in: int,
            n_node_features: int,
            n_hedge_features_out: Optional[int] = None
            ) -> None:

        r"""
        Args:
           key (jax.random.PRNGKey): random key for initialisation purposes.
           n_hedges_features_in (int): input hedge feature vector dimension.
           n_node_features (int): node feature vector dimension.
           n_hedge_features_out (int, optional): output hedge feature vector dimension;
             if not given assumed to be same as n_hedge_features_in 

        """

        key1, key2 = jax.random.split(key, 2)

        if n_hedge_features_out is None:
           n_hedge_features_out = n_hedge_features_in

        # define two linear layers (eventually these may become full MLPs)
        # one for transforming node features, and one for 

        self.hedge_message = eqx.nn.Linear(
             in_features = n_hedge_features_in,
             out_features = n_hedge_features_out,
             key = key1
        )

        self.node_scaling = eqx.nn.Linear(
             in_features = n_node_features,
             out_features = n_hedge_features_out,
             key = key2
        )

    def __call__(self,
            node_features: jnp.ndarray,
            hedge_features: jnp.ndarray,
            hgraph_data: dict[str, jnp.ndarray]   
        ) -> jnp.ndarray:

        r"""
        Args:
           node_features: array of input node features
           hedge_features: array of hedge_feature vectors
           hgraph_data: a pytree dictionary containing HyperGraph index arrays

        Shapes:
           -**inputs:**
           node_features [:, n_node_features], first dimension runs over nodes
           hedge_features [:, n_hedge_features_in], first dimension runs over hedges

           -**outputs:**
           output_hedge_features [:, n_hedge_features_out]

        """

        hedge_senders = hgraph_data['hedge_senders']
        hedge_receivers = hgraph_data['hedge_receivers']
        hedge_adjacency = hgraph_data['hedge_convolution']

        node2hedge_senders = hgraph_data['node2hedge_senders']
        node2hedge_receivers = hgraph_data['node2hedge_receivers']
        node2hedge_convolution = hgraph_data['node2hedge_convolution']

        # first convolve hedge features

        sender_features = hedge_features[hedge_senders]

        # messages = jax.jit(
        #                jax.vmap(self.hedge_message)
        #                   )(sender_features)
        messages = jax.vmap(self.hedge_message)(sender_features)

        scaled_messages = hedge_adjacency * messages

        gathered_messages = jax.ops.segment_sum(scaled_messages,
                               hedge_receivers)

        # now use node embeddings to co-embed node embeddings

        node_sender_features = node_features[node2hedge_senders]

        # node_messages = jax.jit(
        #                     jax.vmap(self.node_scaling)
        #                        )(node_sender_features)
        node_messages = jax.vmap(self.node_scaling)(node_sender_features)

        scaled_node_messages = node2hedge_convolution * node_messages

        gathered_scaling = jax.ops.segment_sum(scaled_node_messages,
                              node2hedge_receivers)

        # output = jax.jit(jnp.tanh)(gathered_scaling * gathered_messages)
        output = jnp.tanh(gathered_scaling * gathered_messages)

        return output

