
from flax import nnx
import jax.numpy as jnp
import jax

from os.path import abspath
import sys

path = abspath('../hypergraph/')

sys.path.append(path)

from hypergraph_batch import hypergraph_batch
from hypergraph_model import HyperGraphConvolution
from QM9_covalent_hypergraphs import QM9CovalentHyperGraphs

import pdb; pdb.set_trace()

file = '/Users/ehe/Work/Databases/QM9DataBase/train/dsgdb9nsd_000001.xyz'

species_list = ['H', 'C', 'N', 'O', 'F']
node_feature_list = ['atomic_number', 'covalent_radius', 'vdw_radius', 'electron_affinity',
                     'en_pauling', 'group']

GM = QM9CovalentHyperGraphs(species_list=species_list,
                            node_feature_list=node_feature_list,
                            n_total_hedge_features=10)

hgraph = GM.structure2graph(file)

_, n_node_features = hgraph.node_features.shape
_, n_hedge_features = hgraph.hedge_features.shape

data_dict = hgraph.data_dict()

# key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(42)

conv_layers = [{'n_node_in': n_node_features,
           'n_hedge_in': n_hedge_features,
           'n_node_out': n_node_features,
           'n_hedge_out': n_hedge_features},
          {'n_node_in': n_node_features,
           'n_hedge_in': n_hedge_features,
           'n_node_out': n_node_features,
           'n_hedge_out': n_hedge_features}]

node_layers = [{'n_node_in': n_node_features,
                'n_node_out': n_node_features},
               {'n_node_in': n_node_features,
                'n_node_out': n_node_features},
               {'n_node_in': n_node_features,
                'n_node_out': 1}]

hedge_layers = [{'n_hedge_in': n_hedge_features},
                {'n_hedge_in': n_hedge_features,
                 'n_hedge_out': 1}]

model = HyperGraphConvolution(
            rngs = rngs,
            conv_layers = conv_layers,
            node_layers = node_layers,
            hedge_layers = hedge_layers
        )

total_energy = model(hgraph)
print(f'Total energy of graph: {total_energy}')

# now create a batch of two graphs
hgraphs = [hgraph, hgraph, hgraph, hgraph]

sgraph = hypergraph_batch(hgraphs)

energies = model(sgraph)

for n, energy in enumerate(energies):
    print(f'Total energy of graph {n}: {energy}')

print('Got here!')
