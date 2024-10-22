
from abc import ABC, abstractmethod
import re

import jax
import jax.numpy as jnp 
import numpy as np
from mendeleev import element

from hypergraph import HyperGraph

class AtomicStructureHyperGraphs(ABC):

    """

    :Class AtomicStructureHyperGraphs:

    This is an Abstract Base Class (abc) that allows easy construction
    of derived classes implementing different strategies to turn
    atomic structural information into a graph. The base class
    implement helper function

    generate_node_features: sets up the arrays of node features for each
          chemical species

    Derived classes need to implement method structure2graph, that
    takes as input an input file containing the structural information
    and returns a torch_geometric.data Data object (graph) constructed
    according to the structure2graph implementation.

    """

    def __init__(
        self,
        species_list: list[str],
        node_feature_list: list[str] = [],
        n_total_hedge_features: int = 10,
        pooling: str = "add"
    ) -> None:

        """

        Initialises an instance of the class. It needs to be passed the list
        of chemical species that may be found in the training files, the
        chemical feature list of each node, which is a list of data
        that Mendeleev can understand (see below), and a total number for
        the node features; besides the chemical/physical features, nodes
        can be assigned initial numerical features that are specific to each
        species and that have no physico-chemical significance (see
        generate_node_features() below for details)

        Args:

        :param species_list list[str]: the chemical symbols of the species present
             in the atomic structures

        :param list[str] node_feature_list: (default empty) contains
             mendeleev data commands to select the required features
             for each feature e.g. node_feature_list = ['atomic_number',
             'atomic_radius', 'covalent_radius']

        :param n_total_hedge_features int: (default = 10) 
             Specifies the total number of hedge features; the first two will be 
             two "physical" features, namely the hedge position (initially set as
             the middle of the bond if the hedge connects two atoms, or the position
             of the atom if the hedge represents an electron in a lone pair), and the
             second will be the radius of the electron sphere. Besides these two "physical"
             features, there will be n_total_hedge_features - 2 additional "unphysical"
             features, put here only to add flexibility to the model, and that will 
             be chosen randomly, initially the same for all hedges.

        :param str pooling: (default = 'add' ) indicates the type of pooling to be
             done over nodes to estimate the fitted property; usually
             this will be 'add', meaning that the property prediction
             is done by summing over nodes; the only other contemplated
             case is 'mean', in which case the prediction is given by
             averaging over nodes. WARNING: this must be done in a
             concerted way (the same) in the GNN model definition!
        
        """

        self.species = species_list

        # the following is a dictionary mapping each chemical symbol to 
        # an integer index

        self.species_dict = {}

        for n, spec in enumerate(self.species):
            self.species_dict[spec] = n 

        self.node_feature_list = node_feature_list

        self.spec_features = self.generate_node_features()

        self.num_node_features = self.spec_features[self.species[0]].size 

        self.n_total_hedge_features = n_total_hedge_features

        n_features = n_total_hedge_features - 4

        # hedge_features are now set up in children classes
        # self.hedge_features = jax.random.normal(key=key, shape=(n_features,))

        self.pooling = pooling

    def generate_node_features(self) -> dict[str, float]:

        """

        This function generates initial node features for an atomic hyper-graph

        This function uses self.node_feature_list to construct a dictionary with
        a key for each species, and values equal to the features. Entries in the 
        node feature list should be valid Mendeleev commands, and then their
        numerical values (e.g. atomic number, covalent radius) are filled in. 
        One exception is: if the command "group" or "period" is used, a
        one-hot encoded array is constructed for the group and period of each 
        species, resulting in a vector of length 25 (7 periods, 18 groups) added
        to whatever other numerical features are given in the node feature list.

        :return: It returns a dictionary where the keys are the chemical symbol and
        the values are an array of dimensions (n_node_features), such that
        initial nodes in the graph can have their features filled according
        to their species. The array of features thus created is later used
        in molecule2graph to generate the corresponding molecular graph.
        :rtype: dict[str, float]

        """

        n_species = len(self.species)

        # generate an element object for each species

        spec_list = []

        for spec in self.species:
            spec_list.append(element(spec))

        # we want to have node features normalised in the range [-1:1]

        use_groupperiod = False

        features = []
        for feature in self.node_feature_list:

            if re.match("^period|^group",feature):

                use_groupperiod = True

            else:

                features.append(feature)

        n_features = len(features)
        values = np.zeros((n_features, n_species), dtype=float)

        for n, spec in enumerate(spec_list):

            for m, feature in enumerate(features):

                command = "spec." + feature
                values[m, n] = eval(command)

        # now detect the maximum and minimum values for each feature
        # over the list of species we have

        features_max = np.zeros(n_features, dtype=float)
        features_min = np.zeros(n_features, dtype=float)

        for m in range(n_features):
            features_max[m] = np.max(values[m, :])
            features_min[m] = np.min(values[m, :])

        # normalise values

        if n_species > 1:

           for n in range(n_species):
               for m in range(n_features):

                   values[m, n] = (
                       2.0
                       * (values[m, n] - features_min[m])
                       / (features_max[m] - features_min[m])
                       - 1.0
                   )

        spec_features = {}

        for n, spec in enumerate(spec_list):

            period = np.zeros(( 7 ), dtype = float )
            period[spec.period-1] = 1.0

            group = np.zeros(( 18 ), dtype = float )
            group[spec.group_id-1] = 1.0

            if use_groupperiod: 
               spec_features[spec.symbol] = \
                     np.concatenate((period, group, values[:, n]))
            else:
               spec_features[spec.symbol] = values[:,n]

        # we are done

        return spec_features

    def n_node_features(self) -> int:
        "return the number of node features"

        return self.num_node_features

    @abstractmethod
    def structure2graph(self, file_name: str) -> HyperGraph:

        """

        This method must be implemented in derived classes. Its purpose
        is to take a file-name as input containing atomic structural
        information and return a HyperGraph object representing the
        same structure.

        Args:

        :param str file_name: path to file containing structure data

        :return: tuple of HyperGraph instance and dictionary of possible targets
        :rtype: HyperGraph (defined in ../hypergraph/hypergraph.py)

        """

        raise NotImplementedError

