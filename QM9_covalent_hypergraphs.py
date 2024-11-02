
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from mendeleev import element
import numpy as np
from scipy.constants import physical_constants

from atomic_structure_hypergraphs import AtomicStructureHyperGraphs
from hypergraph import HyperGraph
# from exceptions import IndexOutOfBounds
from QM9_utils import atom_ref, read_QM9_structure

class QM9CovalentHyperGraphs(AtomicStructureHyperGraphs):

    """

    A class to read molecule information from the QM9 database file
    and convert it to HyperGraph representation. In this class, hyper-graphs
    are constructed in a chemically intuitive way: a node (atom) has connections
    only to other nodes that are at a distance that is up to alpha times the
    sum of their respective covalent radii away, where alpha is
    a factor >= 1 (default 1.1). In this mode edges will correspond
    to chemical bonds. Covalent radii are extracted from Mendeleev for
    each listed species.

    """

    def __init__(
        self,
        species_list: list[str],
        node_feature_list: list[str] = [],
        n_hedge_features: int = 10,
        n_max_neighbours: int = 12,
        pooling: str = "add",
        alpha: float = 1.1,
        r_min: float = 0.5, 
        r_max: float = 2.0,
        shift_energies: bool = True,
    ) -> None:

        # initialise the base class

        super().__init__(
           species_list = species_list, 
           node_feature_list = node_feature_list,
           n_hedge_features = n_hedge_features,
           pooling = pooling
        )

        self.n_max_neighbours = n_max_neighbours
        self.alpha = alpha  # alpha is the scaling factor for bond (edge)
        # critera, i.e. two atoms are bonded if their
        # separation is r <= alpha*(rc1 + rc2), where
        # rci are the respective covalent radii
        self.r_min = r_min
        self.r_max = r_max
        self.shift_energies = shift_energies

        self.covalent_radii = self.get_covalent_radii()

        self.n_valence_electrons = self.get_valence_electrons()

        # define a conversion factor from Hartrees to eV
        self.Hartree2eV = physical_constants["Hartree energy in eV"][0]

    def get_covalent_radii(self) -> dict[str, float]:

        """

        Sets up and returns a dictionary of covalent radii (in Ang)
        for the list of species in its argument

        :return: covalent_radii: dict of covalent radius for eash species (in Angstrom)
        :rtype: dict

        """

        covalent_radii = {}

        for label in self.species:

            spec = element(label)

            covalent_radii[label] = spec.covalent_radius / 100.0
            # mendeleev stores radii in pm, hence the factor

        return covalent_radii

    def get_valence_electrons(self) -> dict[str, int]:

        """
        Set up a dictionary with keys are the chemical species
        and the number of valence electrons as the values

        :return: n_valence_electrons: 
        :rtype: dict

        """

        n_valence_electrons = {}

        for label in self.species:
           
            spec = element(label)

            n_valence_electrons[label] = spec.nvalence()

        return n_valence_electrons

    def structure2graph(self, fileName: str) -> HyperGraph:

        """

        A function to turn atomic structure information imported from
        a  database file and convert it to HyperGraph object. 
        In this particular class graphs are constructed in the following way:

        Chemically intuitive way: initially we build connections between a 
           node (atom) and other nodes that are separated by a distance 
           that is up to alpha times the sum of their respective covalent 
           radii away. In this mode most hyper-edges will correspond to 
           chemical bonds, simply connecting two nodes. There will be 
           two hyper-edges per covalent chemical bond. After all covalent
           connections are built in this way, valence electrons are assigned
           to connections, one electron per connection (two electrons per
           connection), forming hyper-edges. Any remaining un-assigned electrons
           after this process will be assigned to nodes trying to fulfill 
           Lewis's octet rule. 

        Args:

        :param: fileName (string): the path to the file where the structure
           information is stored in file.
        :type: str
        :return: graph representation of the structure contained in fileName
        :rtype: HyperGraph

        """

        (
            molecule_id,
            n_atoms,
            labels,
            positions,
            properties
        ) = read_QM9_structure(fileName, self.shift_energies)

        # the total number of node features is given by the species features

        n_features = self.spec_features[labels[0]].size 

        node_features = np.zeros((n_atoms, n_features), dtype=np.float32)

        # atoms will be hyper-graph nodes; connections will be created for every
        # neighbour of i that is among the nearest
        # n_max_neighbours neighbours of atom i

        # first we loop over all pairs of atoms and calculate the matrix
        # of squared distances

        dij2 = np.zeros((n_atoms, n_atoms), dtype=float)

        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):

                rij = positions[j, :] - positions[i, :]
                rij2 = np.dot(rij, rij)

                dij2[i, j] = rij2
                dij2[j, i] = rij2

        n_neighbours = np.zeros((n_atoms), dtype=int)
        neighbour_distance = np.zeros((n_atoms, self.n_max_neighbours),
                                       dtype=float)
        neighbour_index = np.zeros((n_atoms, self.n_max_neighbours),
                                    dtype=int)

        node0 = []
        node1 = []
        distance = []

        for i in range(n_atoms - 1):
            for j in range(i + 1, n_atoms):

                dcut = self.alpha * (
                    self.covalent_radii[labels[i]] + self.covalent_radii[labels[j]]
                )

                dcut2 = dcut * dcut

                if dij2[i, j] <= dcut2:

                    node0.append(i)
                    node1.append(j)

                    node0.append(j)
                    node1.append(i)

                    dij = np.sqrt(dij2[i, j])

                    distance.append(dij)
                    distance.append(dij)

                    neighbour_distance[i, n_neighbours[i]] = dij
                    neighbour_distance[j, n_neighbours[j]] = dij

                    neighbour_index[i, n_neighbours[i]] = j
                    neighbour_index[j, n_neighbours[j]] = i

                    n_neighbours[i] += 1
                    n_neighbours[j] += 1

                    if n_neighbours[i] == self.n_max_neighbours or \
                       n_neighbours[j] == self.n_max_neighbours:
 
                       raise IndexOutOfBounds("n_max_neighbours {} too \
                          small!!!".format(self.n_max_neighbours))
                       
        edge_index = np.array([node0, node1])

        _, num_edges = edge_index.shape

        # store node features and calculate total number of valence electrons

        n_valence_electrons = 0

        for i in range(n_atoms):

            node_features[i,:] = \
                jnp.array(self.spec_features[labels[i]])

            n_valence_electrons += self.n_valence_electrons[labels[i]]

        # now we can assign valence electrons to edges and check if there are
        # any remaining electrons at the end of the process

        hedges = []
        hnodes = [] 
        hfeatures = []

        for n_edge in range(num_edges): 

            if n_edge % 2 == 0:
               spin = jnp.array([1.])
            else:
               spin = jnp.array([-1.])

            hedges.append(n_edge)
            hedges.append(n_edge) # this is correct, should appear twice
            hnodes.append(edge_index[0,n_edge])
            hnodes.append(edge_index[1,n_edge])

            # here we create the two physical hedge features, namely
            # position and width, and append the "non-physical" features

            node_i, node_j = edge_index[:,n_edge]
            width = distance[n_edge]
            position = (positions[node_i,:] + positions[node_j,:]) / 2.

            # phys = np.concatenate(([width], position))
            # features = jnp.concatenate([jnp.array(phys),self.hedge_features])
            # features = self.hedge_features

            rij = distance[n_edge]

            features = e3nn.soft_one_hot_linspace(
                            rij, 
                            start = self.r_min,
                            end = self.r_max,
                            number = self.n_hedge_features,
                            basis = 'smooth_finite',
                            cutoff = True)
            
            features = jnp.concatenate([spin, features])
            hfeatures.append(features)

        n_remaining_electrons = n_valence_electrons - num_edges

        # here we have to deal with remaining electrons if there are any

        _, node_order = np.unique(hnodes, return_counts=True)
        n_electrons = node_order / 2

        n_edge = num_edges

        while n_remaining_electrons > 0:

           for n in range(n_atoms):

               n_node_electrons = self.n_valence_electrons[labels[n]]

               if int(n_electrons[n]) < n_node_electrons:

                  n_add = n_node_electrons - int(n_electrons[n])

                  for i in range(n_add):

                      hedges.append(n_edge)
                      hnodes.append(n)

                      if n_edge % 2 == 0:
                         spin = jnp.array([1.])
                      else:
                         spin = jnp.array([-1.])

                      width = 1. # default electron width
                      # position = positions[n,:]

                      features = e3nn.soft_one_hot_linspace(
                                      rij, 
                                      start = self.r_min,
                                      end = self.r_max,
                                      number = self.n_hedge_features,
                                      basis = 'smooth_finite',
                                      cutoff = True)

                      features = jnp.concatenate([spin, features])

                      hfeatures.append(features)

                      n_electrons[n] += 1

                      n_edge += 1
                      n_remaining_electrons -= 1
        
        hedge_features = jnp.array(hfeatures)

        # incidence matrix

        incidence = jnp.array([hedges, hnodes])

        # here we calculate the hedge order (number of nodes each hedge
        # connects to), and from it, the weight vector of each hedge 
        # on each node: if hedge is connected to n nodes, its weight
        # on each node it connects to is 1/n

        _, hedge_order = jnp.unique(incidence[0,:], return_counts = True)
        weights = 1. / hedge_order[incidence[0,:]]

        # now we can create a hyper-graph object 

        pos = jnp.array(positions)

        hyper_graph = HyperGraph(
            incidence = incidence,
            node_features = node_features, 
            hedge_features = hedge_features,
            weights = weights,
            targets = properties,
            pos=pos
        )

        return hyper_graph

# register this derived class as subclass of AtomicStructureHyperGraphs

AtomicStructureHyperGraphs.register(QM9CovalentHyperGraphs)

