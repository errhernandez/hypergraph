
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
# import equinox as eqx
from flax import nnx

from convolutions import HedgeConvolution, NodeConvolution
from hypergraph import HyperGraph

class HyperGraphModule(nnx.Module):

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
            rngs: nnx.Rngs,
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

        # key_nodes, key_hedges = jax.random.split(key)

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
            rngs,
            n_node_in,
            n_hedge_in,
            n_node_out
        )

        self.HedgeConv = HedgeConvolution(
            rngs,
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

