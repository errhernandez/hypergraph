
from datetime import datetime
import faulthandler
from glob import glob
import os
import pdb
import re
import sys

from flax import nnx
import jax.numpy as jnp
import jax
import markdown
import matplotlib.pyplot as plt
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import yaml

from hypergraph_batch import hypergraph_batch
from hypergraph_dataset import HyperGraphDataSet
from hypergraph_dataloader import HyperGraphDataLoader

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

# path to the input data

database_path = input_data.get("database", "../Databases/QM9ERHGraphDatabase/")

# the hypergraph description has its own yaml file, so read it

database_file = database_path + '/graph_description.yml'

with open(database_file, 'r') as database_stream:
    database_data = yaml.load(database_stream, Loader=yaml.Loader)

# read training and test data files

train_path = database_path + 'train/'
test_path = database_path + 'test/'

train_files = glob(train_path + '*.pkl')
test_files = glob(test_path + '*.pkl')

n_training_max = input_data.get("n_training_max", None)
n_test_max = input_data.get("n_test_max", None)
train_validation_fraction = input_data.get("train_validation_fraction", 0.3)

if n_training_max is None or n_training_max > len(train_files):
   train_list, valid_list = train_test_split(train_files,
                                 test_size=train_validation_fraction)
else:
   train_list, valid_list = train_test_split(train_files[:n_training_max],
                                 test_size=train_validation_fraction)

if n_test_max is None or n_test_max > len(test_files):
   test_list = test_files
else: 
   test_list = test_files[:n_test_max]
   
# now create training, validation and test datasets

train_dataset = HyperGraphDataSet(files = train_list)
valid_dataset = HyperGraphDataSet(files = valid_list)
test_dataset = HyperGraphDataSet(files = test_list)
   
# read some parameters for optimisation 

n_epochs = input_data.get("n_epochs", 100)
n_batch = input_data.get("n_batch", 50)
n_checkpoint_freq = input_data.get("n_checkpoint_freq", 10)
learning_rate = input_data.get("learning_rate", 1.0e-3)

# create dataloaders for each dataset

train_dl = HyperGraphDataLoader(train_dataset, batch_size = n_batch)
valid_dl = HyperGraphDataLoader(valid_dataset, batch_size = n_batch)
test_dl = HyperGraphDataLoader(test_dataset)

print('Got this far!')

