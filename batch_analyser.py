
from datetime import datetime
import faulthandler
from glob import glob
import os
import pathlib
import pdb
import re
import sys

import equinox as eqx
from flax import nnx
import jax.numpy as jnp
import jax
import markdown
import matplotlib.pyplot as plt
import numpy as np
import optax
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import yaml

from checkpointing import checkpoint_load, checkpoint_save
# from hypergraph_batch import hypergraph_batch
from hypergraph_batching import HyperGraphBatching
from hypergraph_model import HyperGraphConvolution 
from hypergraph_dataset import HyperGraphDataSet
from hypergraph_dataloader import HyperGraphDataLoader
from loss_function import loss_function
from run_banner import run_banner
from training import train_model

"""
A driver script to fit a Graph Convolutional Neural Network GCNN model to
represent properties of molecular/condensed matter systems.

To execute: python fit_model.py input-file

where input-file is a yaml file specifying different parameters of the
model and how the job is to be run. For an example see sample.yml

"""

input_file = sys.argv[1]   # input_file is ayaml compliant file

with open(input_file, 'r') as input_stream:
    input_data = yaml.load(input_stream, Loader=yaml.Loader)

debug = input_data.get("debug", False)

if debug:
    faulthandler.enable()
    pdb.set_trace()

time_string = datetime.now().strftime("%d%m%Y-%H%M%S")

# path to the input data base

database_path = input_data.get("database", "../Databases/QM9ERHGraphDatabase/")

# the hypergraph description has its own yaml file, so read it

database_file = database_path + '/graph_description.yml'

with open(database_file, 'r') as database_stream:
    database_data = yaml.load(database_stream, Loader=yaml.Loader)

# read training and test data files and create train, validation and test sets and loaders

train_path = database_path + 'train/'
test_path = database_path + 'test/'

train_files = glob(train_path + '*.pkl')
test_files = glob(test_path + '*.pkl')

n_training_max = input_data.get("n_training_max", None)
n_test_max = input_data.get("n_test_max", None)
train_validation_fraction = input_data.get("train_validation_fraction", 0.3)

if n_training_max is None or n_training_max > len(train_files):
    train_dataset = HyperGraphDataSet(files = train_files)
else:
    train_dataset = HyperGraphDataSet(files = train_files[:n_training_max])

if n_test_max is None or n_test_max > len(test_files):
   test_list = test_files
else: 
   test_list = test_files[:n_test_max]
   
test_dataset = HyperGraphDataSet(files = test_list)
   
# read some parameters for optimisation 

train_batch_size = input_data.get("training_batch_size", 50)

# create dataloaders for each dataset

train_batching = HyperGraphBatching(
                    dataset = train_dataset,
                    batch_size = train_batch_size 
		)

train_dl = train_batching.dataloader()

n_nodes_batch, n_hedges_batch, n_incidence_batch, \
n_hedge_convolution_batch, n_node_convolution_batch \
   = train_batching.batch_sizes()

test_batching = HyperGraphBatching(
                    dataset = test_dataset,
                    batch_size = 1 
		)

test_dl = test_batching.dataloader()

# now we loop over train batches and check the maximum size of arrays

n_nodes = []
n_hedges = []
n_incidence = []
n_hedge_convolution = []
n_hedge2node_convolution = []
n_node_convolution = []
n_node2hedge_convolution = []
n_node_receivers = []
n_node_senders = []
n_node2hedge_receivers = []
n_node2hedge_senders = []
n_hedge_receivers = []
n_hedge_senders = []
n_hedge2node_receivers = []
n_hedge2node_senders = []

for sample in train_dl:

    n_nodes.append(sample.n_nodes)
    n_hedges.append(sample.n_hedges)
    _, len_inc = sample.incidence.shape
    n_incidence.append(len_inc)
    len_hconv, _ = sample.hedge_convolution.shape
    n_hedge_convolution.append(len_hconv)
    len_h2nconv, _ = sample.hedge2node_convolution.shape
    n_hedge2node_convolution.append(len_h2nconv)
    len_nconv, _ = sample.node_convolution.shape
    n_node_convolution.append(len_nconv)
    len_n2hconv, _ = sample.node2hedge_convolution.shape
    n_node2hedge_convolution.append(len_n2hconv)
    len_node_receivers, = sample.node_receivers.shape
    n_node_receivers.append(len_node_receivers)
    len_nodesenders, = sample.node_senders.shape
    n_node_senders.append(len_nodesenders)
    n_node_receivers.append(len_nodesenders)
    len_node2hedge_receivers, = sample.node2hedge_receivers.shape
    n_node2hedge_receivers.append(len_node2hedge_receivers)
    len_node2hedge_senders, = sample.node2hedge_senders.shape
    n_node2hedge_senders.append(len_node2hedge_senders)
    len_hedge_receivers, = sample.hedge_receivers.shape
    n_hedge_receivers.append(len_hedge_receivers)
    len_hedge_senders, = sample.hedge_senders.shape
    n_hedge_senders.append(len_hedge_senders)
    len_hedge2node_receivers, = sample.hedge2node_receivers.shape
    n_hedge2node_receivers.append(len_hedge2node_receivers)
    len_hedge2node_senders, = sample.hedge2node_senders.shape
    n_hedge2node_senders.append(len_hedge2node_senders)
    
# now convert these to jnp arrays and obtain maximum values

jn_nodes = jnp.array(n_nodes)    
jn_hedges = jnp.array(n_hedges)
jn_incidence = jnp.array(n_incidence)
jn_hedge_convolution = jnp.array(n_hedge_convolution)
jn_hedge2node_convolution = jnp.array(n_hedge2node_convolution)
jn_node_convolution = jnp.array(n_node_convolution) 
jn_node2hedge_convolution = jnp.array(n_node2hedge_convolution)
jn_node_receivers = jnp.array(n_node_receivers)
jn_node_senders = jnp.array(n_node_senders)
jn_node2hedge_receivers = jnp.array(n_node2hedge_receivers)
jn_node2hedge_senders = jnp.array(n_node2hedge_senders)
jn_hedge_receivers = jnp.array(n_hedge_receivers)
jn_hedge_senders = jnp.array(n_hedge_senders)
jn_hedge2node_receivers = jnp.array(n_hedge2node_receivers)
jn_hedge2node_senders = jnp.array(n_hedge2node_senders)

max_nodes = jnp.max(jn_nodes)
max_hedges = jnp.max(jn_hedges)
max_incidence = jnp.max(jn_incidence)
max_hedge_convolution = jnp.max(jn_hedge_convolution)
max_hedge2node_convolution = jnp.max(jn_hedge2node_convolution)
max_node_convolution = jnp.max(jn_node_convolution)
max_node2hedge_convolution = jnp.max(jn_node2hedge_convolution)
max_node_receivers = jnp.max(jn_node_receivers)
max_node_senders = jnp.max(jn_node_senders)
max_node2hedge_receivers = jnp.max(jn_node2hedge_receivers)
max_node2hedge_senders = jnp.max(jn_node2hedge_senders)
max_hedge_receivers = jnp.max(jn_hedge_receivers)
max_hedge_senders = jnp.max(jn_hedge_senders)
max_hedge2node_receivers = jnp.max(jn_hedge2node_receivers)
max_hedge2node_senders = jnp.max(jn_hedge2node_senders)

print(f'max_nodes = {max_nodes}')
print(f'max_hedges = {max_hedges}')
print(f'max_incidence = {max_incidence}')
print(f'max_hedge_convolution = {max_hedge_convolution}')
print(f'max_hedge2node_convolution = {max_hedge2node_convolution}')
print(f'max_node_convolution = {max_node_convolution}')
print(f'max_node2hedge_convolution = {max_node2hedge_convolution}')
print(f'max_node_receivers = {max_node_receivers}')
print(f'max_node_senders = {max_node_senders}')
print(f'max_node2hedge_receivers = {max_node2hedge_receivers}')
print(f'max_node2hedge_senders = {max_node2hedge_senders}')
print(f'max_hedge_receivers = {max_hedge_receivers}')
print(f'max_hedge_senders = {max_hedge_senders}')
print(f'max_hedge2node_receivers = {max_hedge2node_receivers}')
print(f'max_hedge2node_senders = {max_hedge2node_senders}')


print('')
print('')
print(f'n_incidence_batch = {n_incidence_batch}')
print(f'n_hedge_convolution_batch = {n_hedge_convolution_batch}')
print(f'n_node_convolution_batch = {n_node_convolution_batch}')
