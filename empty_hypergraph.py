
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_array

from hypergraph import HyperGraph

def empty_graph(
        n_hedges: int, 
        n_nodes: int,
        n_hedge_features: int,
        n_node_features: int,
        n_max_incidence: int,
        n_max_node_convolution: int, 
        n_max_hedge_convolution: int, 
        sample_targets: dict
    ) -> HyperGraph:

    r""" This function creates an empty HyperGraph used only for 
         padding batches of hypergraphs, so that the batches have 
         all exactly the same dimensions; this is necessary for 
         jitting purposes                 """

    # first create an empty instance and populate it with zeros

    zerograph = HyperGraph(incidence = None)

    zerograph.n_hedges = n_hedges
    zerograph.n_nodes = n_nodes

    zerograph.hedge_features = jnp.zeros((n_hedges, n_hedge_features),
                                          dtype=float)
    zerograph.node_features = jnp.zeros((n_nodes, n_node_features),
                                         dtype=float)

    # here create incidence array; since this is a zero graph, the connectivity
    # between nodes and hedges is arbitrary, so do not waste sleep over this

    incidence0 = []
    incidence1 = []
    
    for i in range(n_max_incidence):

        n_hedge = i % n_hedges
        incidence0.append(n_hedge)

        n_node = i % n_nodes
        incidence1.append(n_node)

    zerograph.incidence = jnp.array([incidence0, incidence1], dtype = int)
    zerograph.weights = jnp.zeros((n_max_incidence), dtype=float)

    # here we create a zero target dictionary using the template in
    # the list of arguments

    targets = {}

    for key in sample_targets.keys():
        value = sample_targets[key]   # value is a list, either of a real number or an array
        if isinstance(value[0], float):
           targets[key] = [0.0]
        elif isinstance(value[0], np.ndarray):
           targets[key] = [np.zeros_like(value[0])]

    zerograph.targets = targets

    zerograph.hedge_convolution = jnp.zeros((n_max_hedge_convolution, 1), \
                                             dtype=float)
    zerograph.node_convolution = jnp.zeros((n_max_node_convolution, 1), \
                                            dtype=float)

    # although it does not change the results (features are all zero), let
    # us use consistent indices

    values = jnp.ones((n_max_incidence), dtype=float)

    Z = csr_array((values, (incidence1, incidence0)),
                  shape=(n_nodes, n_hedges))

    Zt = csr_array((values, (incidence0, incidence1)),
                   shape=(n_hedges, n_nodes))

    Ch = Z.T @ Z
    Cn = Z @ Z.T
    Ch.sort_indices()
    Cn.sort_indices()

    h_send = []
    h_recv = []

    for i in range(n_hedges):

        recv = Ch.indices[Ch.indptr[i]:Ch.indptr[i+1]]
        send = len(recv) * [i]

        [h_recv.append(item) for item in recv]
        [h_send.append(item) for item in send]

    if len(h_recv) > n_max_hedge_convolution:
       h_recv = h_recv[:n_max_hedge_convolution]
    if len(h_send) > n_max_hedge_convolution:
       h_send = h_send[:n_max_hedge_convolution]

    zerograph.hedge_receivers = jnp.array(h_recv)
    zerograph.hedge_senders = jnp.array(h_send)

    n_send = []
    n_recv = []

    for i in range(n_nodes):

        recv = Cn.indices[Cn.indptr[i]:Cn.indptr[i+1]]
        send = len(recv) * [i]

        [n_recv.append(item) for item in recv]
        [n_send.append(item) for item in send]

    if len(n_recv) > n_max_node_convolution:
       n_recv = n_recv[:n_max_node_convolution]
    if len(n_send) > n_max_node_convolution:
       n_send = n_send[:n_max_node_convolution]

    zerograph.node_receivers = jnp.array(n_recv)
    zerograph.node_senders = jnp.array(n_send)

    # the following deals with hedge scaling of nodes and vice versa

    zerograph.node2hedge_convolution = jnp.expand_dims(Z.data, axis=1)
    zerograph.hedge2node_convolution = jnp.expand_dims(Zt.data, axis=1)

    recv_n2h = []
    send_n2h = []

    for i in range(n_nodes):

        recv = Z.indices[Z.indptr[i]:Z.indptr[i+1]]
        send = len(recv) * [i]

        [recv_n2h.append(item) for item in recv]
        [send_n2h.append(item) for item in send]
            
    if len(recv_n2h) > n_max_incidence:
       recv_n2h = recv_n2h[:n_max_incidence]
    if len(send_n2h) > n_max_incidence:
       send_n2h = send_n2h[:n_max_incidence]

    zerograph.node2hedge_receivers = jnp.array(recv_n2h)
    zerograph.node2hedge_senders = jnp.array(send_n2h)

    zerograph.hedge2node_receivers = jnp.array(incidence1)
    zerograph.hedge2node_senders = jnp.array(incidence0)

    zerograph.batch_node_index = jnp.zeros(n_nodes, dtype=int)
    zerograph.batch_hedge_index = jnp.zeros(n_hedges, dtype=int)

    # all done, so return this bogus empty hypergraph

    return zerograph

