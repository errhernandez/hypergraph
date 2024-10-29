
from flax import nnx
import jax.numpy as jnp
import jax

from os.path import abspath
import sys

path = abspath('../hypergraph/')

sys.path.append(path)

from hypergraph_data_loader import HyperGraphDataLoader
from hypergraph_dataset import HyperGraphDataSet
from hypergraph_model import HyperGraphConvolution
from QM9_covalent_hypergraphs import QM9CovalentHyperGraphs

import pdb; pdb.set_trace()

train_path = '/Users/ehe/Work/Databases/QM9ERHGraphDataBase/train/'
test_path = '/Users/ehe/Work/Databases/QM9ERHGraphDataBase/test/'

train_set = HyperGraphDataSet(database_dir = train_path)
test_set = HyperGraphDataSet(database_dir = test_path)

train_dl = HyperGraphDataLoader(dataset = train_set, batch_size=1)
test_dl = HyperGraphDataLoader(dataset = test_set)

# species_list = ['H', 'C', 'N', 'O', 'F']
# node_feature_list = ['atomic_number', 'covalent_radius', 'vdw_radius', 'electron_affinity',
#                    'en_pauling', 'group']

# GM = QM9CovalentHyperGraphs(species_list=species_list,
#                             node_feature_list=node_feature_list,
#                             n_hedge_features=10)

# hgraph = GM.structure2graph(file)

# _, n_node_features = hgraph.node_features.shape
# _, n_hedge_features = hgraph.hedge_features.shape

# indices = hgraph.indices()

# in order to define the model, we need to specify how many node and hedge features these
# hypergraphs have, so get an example and interrogate it

hgraph_sample = train_set.get(0)

_, n_node_features_in = hgraph_sample.node_features.shape
_, n_hedge_features_in = hgraph_sample.hedge_features.shape

# key = jax.random.PRNGKey(42)
rngs = nnx.Rngs(42)

conv_layers = [{'n_node_in': n_node_features_in,
           'n_hedge_in': n_hedge_features_in,
           'n_node_out': 30,
           'n_hedge_out': 30},
          {'n_node_in': 30,
           'n_hedge_in': 30,
           'n_node_out': 30,
           'n_hedge_out': 30}]

node_layers = [{'n_node_in': 30,
                'n_node_out': 30},
               {'n_node_in': 30,
                'n_node_out': 30},
               {'n_node_in': 30,
                'n_node_out': 1}]

hedge_layers = [{'n_hedge_in': 30},
                {'n_hedge_in': 30,
                 'n_hedge_out': 1}]

model = HyperGraphConvolution(
            rngs = rngs,
            conv_layers = conv_layers,
            node_layers = node_layers,
            hedge_layers = hedge_layers
        )

print('Energies of train set')
for n, sample in enumerate(train_dl):
    energy = model(sample)
    print(f'Total energy of graph {n}: {energy}')

print('Energies of test set')
for n, sample in enumerate(test_dl):
    energy = model(sample)
    print(f'Total energy of graph {n}: {energy}')

print('Got here!')
