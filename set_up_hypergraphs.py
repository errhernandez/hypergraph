
import re
from typing import Union

from atomic_structure_hypergraphs import AtomicStructureHyperGraphs
from QM9_covalent_hypergraphs import QM9CovalentHyperGraphs

def set_up_hypergraphs(
    graph_type: str,
    species: list[str],
    node_feature_list: list[str],
    n_total_hedge_features: int = 10,
    n_max_neighbours: int = 12,
    pooling: str = "add",
    **kwargs
) -> AtomicStructureHyperGraphs:

    r"""

    AtomicStructure(Hetero)Graphs factory.

      :param graph_type (str): specifies the type of hypergraph to be constructed: 

          - graph_type = 'QM9': this is the 'chemical' hypergraph rep., in
                  initially connections are established between nodes 
                  separated by a distance 
                  equal or smaller than the sum of covalent radii
                  times alpha, i.e. rij < alpha(rci + rcj). These connections
                  are then turned into hyper-edges.

      :param species list[str]: the list of chemical species seen in the database

      :param node_feature_list list[str]: a list of Mendeleev-recognised keywords identifying
                  chemical species properties (e.g. 'atomic_number', 'covalent_radius', etc).
                  Two special cases of non-Mendeleev keys are accepted, namely 'group' and/or 
                  'period'; if either (or both, but redundant) of these keys is given, then
                  to the list of node features two 1D one-hot encoded vectors will be added, 
                  one of length 7 (with a 1 at the entry corresponding to the element period and
                  zeros elsewhere), and one of length 18 (with a 1 at the entry of the element
                  group). Therefore, using 'group|period' adds 25 features to the nodes (ions). 

      :param n_total_hedge_features int: the number of hedge features

      :param pooling str: the type of pooling to perform by the model, can be 'add' or
          'mean'; the latter is appropriate for energy-per-atom regression, 'add' for 
          total energy regression. 'add' is the default.


    """

    if re.match('^cov', graph_type):

       if 'alpha' in kwargs.keys():
          alpha = kwargs['alpha']
       else:
          alpha = 1.1

       graphs = QM9CovalentHyperGraphs(
                      species_list = species,
                      node_feature_list = node_feature_list,
                      n_total_hedge_features = n_total_hedge_features,
                      n_max_neighbours = n_max_neighbours,
                      pooling = pooling,
                      alpha = alpha 
			    )

    else:  # we make this the default case

       if 'alpha' in kwargs.keys():
          alpha = kwargs['alpha']
       else:
          alpha = 1.1

       graphs = QM9CovalentHyperGraphs(
                      species_list = species,
                      node_feature_list = node_feature_list,
                      n_max_neighbours = n_max_neighbours,
                      pooling = pooling,
                      alpha = alpha 
			    )

    return graphs
