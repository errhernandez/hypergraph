
from typing import Optional, Union

import jax.numpy as jnp
from scipy.sparse import csr_array

class HyperGraph:
    r""" Defines a class to represent a hyper-graph"""

    def __init__(
        self,
        incidence: jnp.ndarray,
        node_features: Optional[jnp.ndarray] = None,
        hedge_features: Optional[jnp.ndarray] = None,
        weights: Optional[jnp.ndarray] = None,
        targets: Optional[jnp.ndarray] = None,
        **kwargs,
    ) -> None:

        self.node_features = node_features
        self.hedge_features = hedge_features
        self.incidence = incidence
        self.weights = weights
        self.targets = targets

        # any additional kw arguments are dealt with below

        for key, value in kwargs.items():
            setattr( self, key, value)

        self.n_hedges = int(max(incidence[0,:])) + 1
        self.n_nodes = int(max(incidence[1,:])) + 1

        if hedge_features is not None:
            n_hedges, n_hedge_features = self.hedge_features.shape
            if n_hedges != self.n_hedges:
                print("Incompatibility between incidence and hedge_features!")
        else:
            n_hedge_features = 0

        self.n_hedge_features = n_hedge_features

        if node_features is not None:
            n_nodes, n_node_features = self.node_features.shape
            if n_nodes != self.n_nodes:
                print("Incompatibility between incidence and node_features!")
        else:
            n_node_features = 0

        self.n_node_features = n_node_features

        # calculate node and hedge order

        _, n_incidence = incidence.shape

        _, self.hedge_order = jnp.unique(incidence[0,:], return_counts=True)
        _, self.node_order = jnp.unique(incidence[1,:], return_counts=True)

        # below calculate the convolution matrices for nodes and hedges 

        _, n_incidence = incidence.shape

        norm = []

        for k in range(n_incidence):

            hedge, node = incidence[:,k]

            value = jnp.sqrt(self.hedge_order[hedge] * self.node_order[node])
            norm.append(1./value)

        values = jnp.array(norm)

        # Z is "normalised" incidence matrix used to construct the convolution
        # matrices for both node-node and hedge-hedge convolution; we store Z
        # as a compressed sparse row (csr) array, and likewise for the conv matrices
        # we also need to create its transpose to have its values in right order
        # as these are needed in the scaling part of the model

        Z = csr_array((values, (incidence[1,:], incidence[0,:])),
                       shape=(self.n_nodes, self.n_hedges))

        Zt = csr_array((values, (incidence[0,:], incidence[1,:])), 
                       shape=(self.n_hedges, self.n_nodes))

        Ch = Z.T @ Z
        Cn = Z @ Z.T
        Ch.sort_indices()
        Cn.sort_indices()

        n_hdata, = Ch.data.shape
        # hdata = Ch.data.reshape((n_hdata,1))
        self.hedge_convolution = jnp.expand_dims(Ch.data, axis=1)
        # self.hedge_convolution = hdata.repeat(self.n_hedge_features, axis=1)
        # above line should not be needed

        n_ndata, = Cn.data.shape
        # ndata = Cn.data.reshape((n_ndata,1))
        self.node_convolution = jnp.expand_dims(Cn.data, axis=1)
        # self.node_convolution = ndata.repeat(self.n_node_features, axis=1)

        h_send = []
        h_recv = []

        for i in range(self.n_hedges):

            recv = Ch.indices[Ch.indptr[i]:Ch.indptr[i+1]]
            send = len(recv) * [i]

            [h_recv.append(item) for item in recv]
            [h_send.append(item) for item in send]

        self.hedge_receivers = jnp.array(h_recv)
        self.hedge_senders = jnp.array(h_send)

        n_send = []
        n_recv = []

        for i in range(self.n_nodes):

            recv = Cn.indices[Cn.indptr[i]:Cn.indptr[i+1]]
            send = len(recv) * [i]

            [n_recv.append(item) for item in recv]
            [n_send.append(item) for item in send]

        self.node_receivers = jnp.array(n_recv)
        self.node_senders = jnp.array(n_send)

        # the following deals with hedge scaling of nodes and vice versa

        self.node2hedge_convolution = jnp.expand_dims(Z.data, axis=1)
        self.hedge2node_convolution = jnp.expand_dims((Z.T).data, axis=1)

        recv_n2h = []
        send_n2h = []

        for i in range(self.n_nodes):

            recv = Z.indices[Z.indptr[i]:Z.indptr[i+1]]
            send = len(recv) * [i]

            [recv_n2h.append(item) for item in recv]
            [send_n2h.append(item) for item in send]
            
        self.node2hedge_receivers = jnp.array(recv_n2h)
        self.node2hedge_senders = jnp.array(send_n2h)

        self.hedge2node_receivers = jnp.array(incidence[1,:])
        self.hedge2node_senders = jnp.array(incidence[0,:])

