
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import equinox as eqx

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
            key: jnp.ndarray,
            n_node_features_in: int,
            n_hedge_features: int,
            n_node_features_out: Optional[int] = None
            ) -> None:

        r"""
        Args:
           key (jax.random.PRNGKey): random key for initialisation purposes.
           n_node_features_in (int): input node feature vector dimension.
           n_hedge_features (int): hedge feature vector dimension.
           n_node_features_out (int, optional): output node feature vector dimension;
             if not given assumed to be same as n_node_features_in

        """

        if n_node_features_out is None: n_node_features_out = n_node_features_in

        # define two linear layers (eventually these may become full MLPs)
        # one for transforming node features, and one for obtaining
        # hedge scaling  

        key1, key2 = jax.random.split(key)

        self.node_message = \
            eqx.nn.Linear(n_node_features_in, n_node_features_out, key=key1)

        # self.node_message = jax.vmap(node_lin)

        self.hedge_scaling = \
            eqx.nn.Linear(n_hedge_features, n_node_features_out, key=key2)

        # self.hedge_scaling = jax.vmap(hedge_lin)

    def __call__(self,
            node_features: jnp.ndarray,
            hedge_features: jnp.ndarray,
            hgraph_data: Dict[str, jnp.ndarray]
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

        messages = jax.vmap(self.node_message)(sender_features)

        scaled_messages = node_convolution * messages

        gathered_messages = jax.ops.segment_sum(scaled_messages,
                               node_receivers)

        # now use hedge embeddings to co-embed node embeddings

        hedge_sender_features = hedge_features[hedge2node_senders]

        hedge_messages = jax.vmap(self.hedge_scaling)(hedge_sender_features)

        scaled_hedge_messages = hedge2node_convolution * hedge_messages

        gathered_scaling = jax.ops.segment_sum(scaled_hedge_messages,
                              hedge2node_receivers)

        output = gathered_scaling * gathered_messages

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
            key: jnp.ndarray,
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

        if n_hedge_features_out is None:
           n_hedge_features_out = n_hedge_features_in

        # define two linear layers (eventually these may become full MLPs)
        # one for transforming node features, and one for 

        key1, key2 = jax.random.split(key)

        self.hedge_message = \
            eqx.nn.Linear(n_hedge_features_in, n_hedge_features_out, key=key1)

        self.node_scaling = \
            eqx.nn.Linear(n_node_features, n_hedge_features_out, key=key2)

    def __call__(self,
            node_features: jnp.ndarray,
            hedge_features: jnp.ndarray,
            hgraph_data: Dict[str, jnp.ndarray]   
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

        messages = jax.vmap(self.hedge_message)(sender_features)

        scaled_messages = hedge_adjacency * messages

        gathered_messages = jax.ops.segment_sum(scaled_messages,
                               hedge_receivers)

        # now use node embeddings to co-embed node embeddings

        node_sender_features = node_features[node2hedge_senders]

        node_messages = jax.vmap(self.node_scaling)(node_sender_features)

        scaled_node_messages = node2hedge_convolution * node_messages

        gathered_scaling = jax.ops.segment_sum(scaled_node_messages,
                              node2hedge_receivers)

        output = gathered_scaling * gathered_messages

        return output

class HyperGraphModule(eqx.Module):

    r"""
    This module represents a layer of HyperGraph node-hedge combined convolution, in 
    which first nodes are convoluted with themselves with a weight factor depending on
    their incident hedged (see NodeConvolution above for details); subsequently hedges 
    are themselves convoluted with neighbouring hedges, and weighted by the just updated
    node features (see HedgeConvolution for details). Node convolution is implemented
    in module NodeConvolution, and hedge convolution in module HedgeConvolution, both 
    given above. 

    """

    NodeConv: NodeConvolution
    HedgeConv: HedgeConvolution

    def __init__(self,
            key: jnp.ndarray,
            n_node_in: int,
            n_hedge_in: int,
            n_node_out: Optional[int] = None,
            n_hedge_out: Optional[int] = None,
            activation: Optional = None
        ) -> None:

        r"""
        Args:
           key (jax random key: initialization key
           n_node_in (int): input node features size
           n_hedge_in (int): input hedge features size
           n_node_out (int): output node embeddings size
           n_hedge_out (int): output hedge embeddings size
           activation: element-wise activation function, default = jnp.tanh
        """

        key_nodes, key_hedges = jax.random.split(key)

        if n_node_out is None: n_node_out = n_node_in
        if n_hedge_out is None: n_hedge_out = n_hedge_in

        # if activation is None:
        #  setattr( self, activation, (jnp.tanh()) )
        # else:
        #   setattr( self, activation, activation )
        # 
        # must find out how to pass optional activation function!
        # for the moment this is hard-coded as jnp.tanh()

        self.NodeConv = NodeConvolution(
            key_nodes,
            n_node_in,
            n_hedge_in,
            n_node_out
        )

        self.HedgeConv = HedgeConvolution(
            key_hedges,
            n_hedge_in,
            n_node_out,
            n_hedge_out
        ) 

    def __call__(self,
            node_features: jnp.ndarray,
            hedge_features: jnp.ndarray,
            hgraph_data: Dict[str, jnp.ndarray]
        ) -> Tuple[jnp.ndarray, jnp.ndarray]: 

        r"""
        Args:
           node_features (jnp.ndarray): input node features
           hedge_features (jnp.ndarray): input hedge features
           hgraph_data (Dict[str, jnp.ndarray]): pytree dictionary with hypergraph
             information indices

        Shapes:
           -**inputs:**
           node_features [:, n_node_features_in], first dimension runs over
             nodes in hypergraph (batch)
           hedge_features [:, n_hedge_features_in], first dimension runs over
             hedges in hypergraph (batch)

           -**outputs:**
           Tuple[out_node_features[:, n_node_features_out],
                 out_hedge_features[:, n_hedge_features_out]]

        """

        # first convolve nodes

        c_node_features = self.NodeConv(
            node_features,
            hedge_features,
            hgraph_data
        )

        new_node_features = jnp.tanh(c_node_features)

        # then convolve hedges
 
        c_hedge_features = self.HedgeConv(
            new_node_features,
            hedge_features,
            hgraph_data
        )

        new_hedge_features = jnp.tanh(c_hedge_features)

        return new_node_features, new_hedge_features

