
import e3nn_jax as e3nn
import jax
import jax.numpy as jnp
from mendeleev import element
import numpy as np
from scipy.constants import physical_constants

from atomic_structure_hypergraphs import AtomicStructureHyperGraphs
from hypergraph import HyperGraph
# from exceptions import IndexOutOfBounds
from EFF_utils import read_EFF_structure

class EFFHyperGraphs(AtomicStructureHyperGraphs):

    """

    A class to read molecule information from the data provided by Anthony Mannino,
    representing molecules consisting of atoms and e-balls, the latter being 
    specified by their Cartesian positions and a width,
    and convert it to HyperGraph representation. In this type of hyper-graph
    e-balls play the role of edges. An e-ball is connected to an atom (node) if
    the sum of the covalent radius of the atom species and the e-ball width is
    less than the distance from the atom to the e-ball centre. In this way we
    construct the incidence matrix and from it the hypergraph representation.
    

    """

    def __init__(
        self,
        species_list: list[str],
        node_feature_list: list[str] = [],
        n_hedge_features: int = 10,
        pooling: str = "add",
        alpha: float = 1.2,
        r_min: float = 0.5, 
        r_max: float = 2.0,
        shift_energies: bool = False,
    ) -> None:

        # initialise the base class

        super().__init__(
           species_list = species_list, 
           node_feature_list = node_feature_list,
           n_hedge_features = n_hedge_features,
           pooling = pooling
        )

        self.alpha = alpha  # alpha is the scaling factor for bond (edge)
        # critera, i.e. two atoms are bonded if their
        # separation is r <= alpha*(rc1 + rc2), where
        # rci are the respective covalent radii
        self.r_min = r_min
        self.r_max = r_max
        self.shift_energies = shift_energies

        # self.covalent_radii = self.get_covalent_radii()

        # self.n_valence_electrons = self.get_valence_electrons()

        # define a conversion factor from Hartrees to eV
        self.Hartree2eV = physical_constants["Hartree energy in eV"][0]

    def structure2graph(self, fileName: str) -> HyperGraph:

        """

        A function to turn atomic structure information imported from
        a  database file and convert it to HyperGraph object. 
        In this particular class graphs are constructed in the following way:

        The structure consists of atoms (ions) and e-balls, electron blobs specified
        by a Cartesian position and a radius (they are assumed spherical). e-balls play
        the role of edges, linking atoms when two atoms are connected to the same e-ball.
        The criterium for this is that an atom is assumed "bound" to an e-ball when 
        the distance from the atom to the e-ball position is less than the sum of the
        e-ball radius and the atom's covalent radius. 

        Args:

        :param: fileName (string): the path to the file where the structure
           information is stored in file.
        :type: str
        :return: graph representation of the structure contained in fileName
        :rtype: HyperGraph

        """

        molecule = read_EFF_structure(fileName)

        # molecule is a dict storing all information on the molecule

        w_radii = np.array(molecule['w_radii'])
        n_wc, = w_radii.shape
        w_centers = np.array(molecule['w_centers']).reshape((n_wc, 3))
        w_spin = np.array(molecule['w_spin']).astype(int)

        za = np.array(molecule['za']).astype(int)
        n_atoms, = za.shape
        ra = np.array(molecule['ra']).reshape((n_atoms, 3))

        try:
          dipole = np.array(molecule['dipole'])
        except:
          dipole = np.zeros((3))
        try:
          quadrupole = np.array(molecule['quadrupole']).reshape((3,3))
        except:
          quadrupole = np.zeros((3,3))

        # polarizability = molecule['polarizability'][0]  # not all molecules have this!

        vibrations = np.array(molecule['vibrations'])
        formationEnthalpy0K = molecule['formationEnthalpy0K'][0]
        # ionizationEnergy = molecule['ionizationEnergy'][0] # not all molecules have this!
        spinPolarization = int(molecule['spinPolarization'])

        properties = {}
        properties['dipole'] = dipole
        properties['dipole'] = np.zeros((3))
        properties['quadrupole'] = quadrupole
        properties['vibrations'] = vibrations
        properties['formationEnthalpy0K'] = formationEnthalpy0K
        properties['spinPolarization'] = spinPolarization
        try:
          properties['charge'] = molecule['charge']
        except:
          properties['charge'] = 0

        # create a list of elements in the molecule

        unique_za, index_za = np.unique(za, return_inverse=True)

        unique_za = unique_za.tolist()

        elements = []

        for atomic_number in unique_za:
            elements.append(element(atomic_number))

        labels = []
        for n, atomic_number in enumerate(za):
            labels.append(elements[index_za[n]].symbol)

        # here we construct the connectivity between Wannier centers and ions

        edges = []
        ions = []

        for i in range(n_wc):  # loop over Wannier centres

            re = w_centers[i,:]
            w_radius = w_radii[i]

            for j in range(n_atoms):

                cov_radius = elements[index_za[j]].covalent_radius * 0.01
                                # we divide by 100 because Mendeleev 
                                # stores values in pm

                rj = ra[j,:]
                rej = rj - re
                dej = np.sqrt(np.dot(rej, rej))

                # if the sum of the covalent radius of the atom and the e-ball width
                # is larger than the distance from atom to e-ball centre, check!

                if dej <= cov_radius + w_radius:

                   edges.append(i)
                   ions.append(j)

        incidence = jnp.array([edges, ions])

        # the total number of node features is given by the species features

        n_features = self.spec_features[labels[0]].size 

        node_features = np.zeros((n_atoms, n_features), dtype=np.float32)

        # store node features and calculate total number of valence electrons

        for i in range(n_atoms):

            node_features[i,:] = \
                jnp.array(self.spec_features[labels[i]])

        hedges = []
        hnodes = [] 
        hfeatures = []

        for n_edge in range(n_wc): 

            spin = jnp.array([w_spin[n_edge]])

            # hedges.append(n_edge)
            # hedges.append(n_edge) # this is correct, should appear twice

            # node_i, node_j = edge_index[:,n_edge]
           
            # hnodes.append(node_i)
            # hnodes.append(node_j)

            # here we create the two physical hedge features, namely
            # position and width, and append the "non-physical" features

            width = w_radii[n_edge]
            position = w_centers[n_edge,:]

            # phys = np.concatenate(([width], position))
            # features = jnp.concatenate([jnp.array(phys),self.hedge_features])
            # features = self.hedge_features

            w_features = e3nn.soft_one_hot_linspace(
                            width, 
                            start = self.r_min,
                            end = self.r_max,
                            number = self.n_hedge_features,
                            basis = 'smooth_finite',
                            cutoff = True)

            # p_features = e3nn. here we need to do something about positions
            
            features = jnp.concatenate([spin, w_features])
            hfeatures.append(features)

        hedge_features = jnp.array(hfeatures)

        # here we calculate the hedge order (number of nodes each hedge
        # connects to), and from it, the weight vector of each hedge 
        # on each node: if hedge is connected to n nodes, its weight
        # on each node it connects to is 1/n

        _, hedge_order = jnp.unique(incidence[0,:], return_counts = True)
        weights = 1. / hedge_order[incidence[0,:]]

        # now we can create a hyper-graph object 

        hyper_graph = HyperGraph(
            incidence = incidence,
            node_features = node_features, 
            hedge_features = hedge_features,
            weights = weights,
            targets = properties,
            pos=ra
        )

        return hyper_graph

# register this derived class as subclass of AtomicStructureHyperGraphs

AtomicStructureHyperGraphs.register(EFFHyperGraphs)

