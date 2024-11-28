
from typing import Tuple

from glob import glob
from math import ceil

import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_array
from torch.utils.data import DataLoader

from hypergraph import HyperGraph
from hypergraph_dataloader import HyperGraphDataLoader
from hypergraph_dataset import HyperGraphDataSet

class SmallParameter(Exception):
    pass

class HyperGraphBatching:

    r""" 

    A class to create batches of a given number of HyperGraph objects.
    When an object of this class is constructed, it first loops through
    it first creates a preliminary list of batches, and over that list
    it calculates the maximum number of nodes and hedges per batch, rounding 
    upwards to the nearest 10, or 100 (see arguments below); the last graph
    in the batch consists of the remaining (empty) nodes and hedges to make 
    fill up to the maximum number of nodes and hedges, padding with zeros
    the corresponding arrays. In this way, all batches will have exactly
    the same size (albeit different number of real nodes and hedges), so
    we can use a stateful (and thus jit-compilable) number of segments
    in the various segment_sum operations in the HyperGraphModel.

    """

    def __init__(
        self,
        dataset: HyperGraphDataSet, 
        batch_size: int = 1,
        drop_last: bool = False
	) -> None:

        """

        Args:

        :param HyperGraphDataSet dataset: 

        :param int batch_size: The number of hypergraphs per batch

        :param bool drop_last: if true, the last batch, smaller than the
               rest if len(self.dataset) % batch_size != 0, is omitted from the list

        """

        self.dataset = dataset
        
        self.batch_size = batch_size

        self.n_total_batches = int(len(self.dataset)/self.batch_size)

        if len(self.dataset) % self.batch_size > 0 and not drop_last:
           self.n_total_batches += 1

        # below we are going to determine the maximum number of nodes and hedges
        # in batches of size self.batch_size

        n_max_nodes = 0
        n_max_hedges = 0

        item = 0

        for n in range(self.n_total_batches):

            n_nodes = 0
            n_hedges = 0

            for m in range(self.batch_size):

                hgraph = self.dataset[item]

                n_nodes += hgraph.n_nodes
                n_hedges += hgraph.n_hedges

                item += 1

            if n_nodes > n_max_nodes: n_max_nodes = n_nodes
     
            if n_hedges > n_max_hedges: n_max_hedges = n_hedges

        # we are going to round up the maximum number of nodes to the 
        # next ten upwards; the maximum number of hedges is rounded
        # up to the next hundred upwards

        self.n_max_nodes = (int(n_max_nodes/10) + 1) * 10
        self.n_max_hedges = (int(n_max_hedges/100) + 1) * 100

    def batch_sizes(self) -> Tuple[int, int]:
        r'Return n_max_nodes, n_max_hedges'

        return self.n_max_nodes, self.n_max_hedges

    def dataloader(self) -> HyperGraphDataLoader:

        return DataLoader(
                   dataset = self.dataset,
                   batch_size = self.batch_size, 
                   collate_fn = self.hypergraph_batch
	       )

    def hypergraph_batch(self, hypergraphs: list[HyperGraph]) -> HyperGraph:

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

        # we will take the pattern of the targets of these nodes to generate
        # an empty target dict for the padding empty-hypergraph at the end

        sample_targets = hypergraphs[0].targets
        empty_targets = {}

        for key in sample_targets.keys():
           empty_targets[key] = [0.0 for n in range(len(sample_targets[key]))]

        node_index = []
        hedge_index = []
        # batch_node_index and batch_hedge_index are 1d vectors that map nodes an
        # hedges (respectively) in a super-graph to the original hypergraph
        # from which they came; it allows to map individual graphs
        # to the right target when fitting the model

        n_hedges = 0
        n_nodes = 0

        for n_graph, hgraph in enumerate(hypergraphs):

            node_index += hgraph.n_nodes*[n_graph]
            hedge_index += hgraph.n_hedges*[n_graph]

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

            if n_hedges > self.n_max_hedges:
               raise SmallParameter('n_hedges', n_hedges, self.n_max_hedges)

            if n_nodes > self.n_max_nodes:
               raise SmallParameter('n_nodes', n_nodes, self.n_max_noes)

        # note that the last graph in the batch is an empty graph, which has 
        # a number of nodes equal to self.n_max_nodes - n_nodes and a number
        # of hedges equal to self.n_max_hedges - n_hedges; for this we create
        # an appropriately sized HyperGraph instance with zeros for hedge and 
        # node features

        _, n_node_features = hypergraphs[0].node_features.shape
        _, n_hedge_features = hypergraphs[0].hedge_features.shape

        n_last_nodes = self.n_max_nodes - n_nodes
        n_last_hedges = self.n_max_hedges - n_hedges

        empty_node_features = jnp.zeros((n_last_nodes, n_node_features))
        empty_hedge_features = jnp.zeros((n_last_hedges, n_hedge_features))

        empty_weights = jnp.zeros((n_last_hedges))

        # create a bogus incidence matrix

        incidence0 = [n for n in range(n_last_hedges)]
        incidence1 = []

        for n in range(int(n_last_hedges/n_last_nodes)):
            for m in range(n_last_nodes):
                incidence1.append(m)

        for n in range(n_last_hedges % n_last_nodes):
            incidence1.append(n)

        empty_incidence = jnp.array([incidence0, incidence1])

        empty_graph = HyperGraph(
                         incidence = empty_incidence,
                         node_features = empty_node_features,
                         hedge_features = empty_hedge_features, 
                         weights = empty_weights
                                )

        # now add this empty hypergraph to the batch
 
        node_index += empty_graph.n_nodes*[n_graph+1]
        hedge_index += empty_graph.n_hedges*[n_graph+1]

        hedges = np.asarray(empty_graph.incidence[0,:]) + n_hedges
        b_hedges = np.concatenate((b_hedges, hedges))

        hedge_receivers = np.asarray(empty_graph.hedge_receivers) + n_hedges
        b_hedge_receivers = np.concatenate((b_hedge_receivers,
                                                    hedge_receivers))
        hedge_senders = np.asarray(empty_graph.hedge_senders) + n_hedges
        b_hedge_senders = np.concatenate((b_hedge_senders, hedge_senders))
        hedge2node_receivers = np.asarray(empty_graph.hedge2node_receivers) + \
                                   n_nodes
        b_hedge2node_receivers = np.concatenate((b_hedge2node_receivers,
                                                     hedge2node_receivers))
        hedge2node_senders = np.asarray(empty_graph.hedge2node_senders) + n_hedges
        b_hedge2node_senders = np.concatenate((b_hedge2node_senders,
                                                   hedge2node_senders))

        nodes = np.asarray(empty_graph.incidence[1,:]) + n_nodes
        b_nodes = np.concatenate((b_nodes, nodes))

        node_receivers = np.asarray(empty_graph.node_receivers) + n_nodes
        b_node_receivers = np.concatenate((b_node_receivers, node_receivers))
        node_senders = np.asarray(empty_graph.node_senders) + n_nodes
        b_node_senders = np.concatenate((b_node_senders, node_senders))
        node2hedge_receivers = np.asarray(empty_graph.node2hedge_receivers) + \
                                     n_hedges
        b_node2hedge_receivers = np.concatenate((b_node2hedge_receivers,
                                                     node2hedge_receivers))
        node2hedge_senders = np.asarray(empty_graph.node2hedge_senders) + n_nodes
        b_node2hedge_senders = np.concatenate((b_node2hedge_senders,
                                                   node2hedge_senders))

        # these must be concatenated at the end of the loop
        node_features.append(empty_graph.node_features)
        hedge_features.append(empty_graph.hedge_features)
        hedge_convolution.append(empty_graph.hedge_convolution)
        hedge2node_convolution.append(empty_graph.hedge2node_convolution)
        node_convolution.append(empty_graph.node_convolution)
        node2hedge_convolution.append(empty_graph.node2hedge_convolution)
        weights.append(empty_graph.weights)

        for key in sample_targets.keys():
            value = empty_targets[key]
            batch_value = targets.get(key, [])
            batch_value.append(value)
            targets[key] = batch_value

        n_hedges += empty_graph.n_hedges
        n_nodes += empty_graph.n_nodes

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

