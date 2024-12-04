
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_array

from hypergraph import HyperGraph

def hypergraph_batch(hypergraphs: list[HyperGraph]) -> HyperGraph:

    r""" 

    This function takes as input a list of Hypergraph instances, and 
    contatenates them into a single HyperGraph instance for batching purposes. 

    It is assumed that all hypergraphs in the list are comparable, in the sense
    that node and hedge features are of the same length and type. It is the 
    responsibility of the caller to ensure that this is the case.

    """

    n_hgraphs = len(hypergraphs)

    # we loop over each hypergraph in the list and create new
    # lists for creating the corresponding super-hypergraph with
    # correlative node and hedge indices

    b_hedges = np.array([], dtype=np.int32)
    b_hedge_receivers = np.array([], np.int32)
    b_hedge_senders = np.array([], np.int32)
    b_hedge2node_receivers = np.array([], dtype=np.int32)
    b_hedge2node_senders = np.array([], dtype=np.int32)

    b_nodes = np.array([], dtype=np.int32)
    b_node_receivers = np.array([], dtype=np.int32)
    b_node_senders = np.array([], dtype=np.int32)
    b_node2hedge_receivers = np.array([], dtype=np.int32)
    b_node2hedge_senders = np.array([], dtype=np.int32)

    node_features = []
    hedge_features = []
    hedge_convolution = []
    hedge2node_convolution = []
    node_convolution = []
    node2hedge_convolution = []
    weights = []

    # targets is a dictionary with the same keys as individual hypergraphs 
    # (assumed to have all the same keys); the values for a batch of hypergraphs
    # will be arrays of the same length as the batch
    targets = {}

    node_index = []
    hedge_index = []
    # batch_node_index and batch_hedge_index are 1d vectors that map nodes an
    # hedges (respectively) in a super-graph to the original hypergraph
    # from which they came; it allows to map individual graphs
    # to the right target when fitting the model

    n_hedges = 0
    n_nodes = 0

    for n_graph, hgraph in enumerate(hypergraphs):

        node_index += int(hgraph.n_nodes) * [n_graph]
        hedge_index += int(hgraph.n_hedges) * [n_graph]

        hedges = np.asarray(hgraph.incidence[0,:]) + n_hedges
        b_hedges = np.concatenate((b_hedges, hedges))

        hedge_receivers = np.asarray(hgraph.hedge_receivers) + n_hedges
        b_hedge_receivers = np.concatenate((b_hedge_receivers,
                                                hedge_receivers))
        hedge_senders = np.asarray(hgraph.hedge_senders) + n_hedges
        b_hedge_senders = np.concatenate((b_hedge_senders, hedge_senders))
        hedge2node_receivers = np.asarray(hgraph.hedge2node_receivers) + \
                               n_nodes
        b_hedge2node_receivers = np.concatenate((b_hedge2node_receivers,
                                                 hedge2node_receivers))
        hedge2node_senders = np.asarray(hgraph.hedge2node_senders) + n_hedges
        b_hedge2node_senders = np.concatenate((b_hedge2node_senders,
                                               hedge2node_senders))

        nodes = np.asarray(hgraph.incidence[1,:]) + n_nodes
        b_nodes = np.concatenate((b_nodes, nodes))

        node_receivers = np.asarray(hgraph.node_receivers) + n_nodes
        b_node_receivers = np.concatenate((b_node_receivers, node_receivers))
        node_senders = np.asarray(hgraph.node_senders) + n_nodes
        b_node_senders = np.concatenate((b_node_senders, node_senders))
        node2hedge_receivers = np.asarray(hgraph.node2hedge_receivers) + \
                                 n_hedges
        b_node2hedge_receivers = np.concatenate((b_node2hedge_receivers,
                                                 node2hedge_receivers))
        node2hedge_senders = np.asarray(hgraph.node2hedge_senders) + n_nodes
        b_node2hedge_senders = np.concatenate((b_node2hedge_senders,
                                               node2hedge_senders))

        # these must be concatenated at the end of the loop
        node_features.append(hgraph.node_features)
        hedge_features.append(hgraph.hedge_features)
        hedge_convolution.append(hgraph.hedge_convolution)
        hedge2node_convolution.append(hgraph.hedge2node_convolution)
        node_convolution.append(hgraph.node_convolution)
        node2hedge_convolution.append(hgraph.node2hedge_convolution)
        weights.append(hgraph.weights)

        for key in hgraph.targets.keys():
            value = hgraph.targets[key]
            batch_value = targets.get(key, [])
            batch_value.append(value)
            targets[key] = batch_value

        n_hedges += hgraph.n_hedges
        n_nodes += hgraph.n_nodes

    # now we can create a batch hgraph (concatenation of individual hgraphs)
    # with the accummulated arrays

    # create an "empty" hypergraph and populate it

    batch_hgraph = HyperGraph(incidence = None)

    batch_hgraph.n_hedges = n_hedges
    batch_hgraph.n_nodes = n_nodes

    batch_hgraph.batch_node_index = jnp.array(node_index)
    batch_hgraph.batch_hedge_index = jnp.array(hedge_index)

    incidence = jnp.array([b_hedges, b_nodes])
    batch_hgraph.incidence = incidence

    batch_hgraph.node_features = jnp.concatenate(node_features)
    batch_hgraph.hedge_features = jnp.concatenate(hedge_features)
    batch_hgraph.hedge_convolution = jnp.concatenate(hedge_convolution)
    batch_hgraph.hedge2node_convolution = jnp.concatenate(hedge2node_convolution)
    batch_hgraph.node_convolution = jnp.concatenate(node_convolution)
    batch_hgraph.node2hedge_convolution = jnp.concatenate(node2hedge_convolution)
    batch_hgraph.weights = jnp.concatenate(weights)
    batch_hgraph.targets = targets

    batch_hgraph.node_receivers = jnp.array(b_node_receivers)
    batch_hgraph.node_senders = jnp.array(b_node_senders)
    batch_hgraph.node2hedge_receivers = jnp.array(b_node2hedge_receivers)
    batch_hgraph.node2hedge_senders = jnp.array(b_node2hedge_senders)

    batch_hgraph.hedge_receivers = jnp.array(b_hedge_receivers)
    batch_hgraph.hedge_senders = jnp.array(b_hedge_senders)
    batch_hgraph.hedge2node_receivers = jnp.array(b_hedge2node_receivers)
    batch_hgraph.hedge2node_senders = jnp.array(b_hedge2node_senders)

    return batch_hgraph
