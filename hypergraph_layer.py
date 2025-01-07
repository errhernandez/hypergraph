
from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from convolutions import HedgeConvolution, NodeConvolution
from hypergraph import HyperGraph

class HyperGraphLayer(eqx.Module):

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
            key: jax.Array,
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
            key = key_nodes,
            n_node_features_in = n_node_in,
            n_hedge_features = n_hedge_in,
            n_node_features_out = n_node_out
        )

        self.HedgeConv = HedgeConvolution(
            key = key_hedges,
            n_hedge_features_in = n_hedge_in,
            n_node_features = n_node_out,
            n_hedge_features_out = n_hedge_out
        ) 

    def __call__(self,
            node_features: jnp.ndarray,
            hedge_features: jnp.ndarray,
            hgraph_data: dict[str, jnp.ndarray]
        ) -> tuple[jnp.ndarray, jnp.ndarray]: 

        r"""
        Args:
           node_features (jnp.ndarray): input node features
           hedge_features (jnp.ndarray): input hedge features
           hgraph_data (dict[str, jnp.ndarray]): pytree dictionary with hypergraph
             information indices

        Shapes:
           -**inputs:**
           node_features [:, n_node_features_in], first dimension runs over
             nodes in hypergraph (batch)
           hedge_features [:, n_hedge_features_in], first dimension runs over
             hedges in hypergraph (batch)

           -**outputs:**
           tuple[out_node_features[:, n_node_features_out],
                 out_hedge_features[:, n_hedge_features_out]]

        """

        # first convolve nodes

        new_node_features = self.NodeConv(
            node_features,
            hedge_features,
            hgraph_data
        )

        # then convolve hedges
 
        new_hedge_features = self.HedgeConv(
            new_node_features,
            hedge_features,
            hgraph_data
        )

        return new_node_features, new_hedge_features

