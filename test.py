
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

layers = [{'n_node_in': n_node_features,
           'n_hedge_in': n_hedge_features,
           'n_node_out': n_node_features+5,
           'n_hedges_out': n_hedge_features+3},
          {'n_node_in': n_node_features+5,
           'n_hedge_in': n_hedge_features+3,
           'n_node_out': n_node_features-10,
           'n_hedge_out': n_hedge_features-8}]

model = HyperGraphConvolution(
            rngs = rngs,
            layers = layers
        )

new_node_features, new_hedge_features = model(hgraph)

# now create a batch of two graphs
hgraphs = [hgraph, hgraph]

sgraph = hypergraph_batch(hgraphs)

new_sgraph_nodes, new_sgraph_hedges = model(sgraph)

n_nodes = hgraph.n_nodes
n_hedges = hgraph.n_hedges
print(jnp.allclose(new_node_features, new_sgraph_nodes[0:n_nodes,:]))
print(jnp.allclose(new_node_features, new_sgraph_nodes[n_nodes:,:]))
print(jnp.allclose(new_hedge_features, new_sgraph_hedges[0:n_hedges,:]))
print(jnp.allclose(new_hedge_features, new_sgraph_hedges[n_hedges:,:]))

print('Got here!')
