
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
import optax
import pickle
from sklearn.model_selection import train_test_split
import yaml

from hypergraph_batch import hypergraph_batch
from hypergraph_model import HyperGraphConvolution 
from hypergraph_dataset import HyperGraphDataSet
from hypergraph_dataloader import HyperGraphDataLoader
from loss_function import loss_function
from training import train_step, eval_step

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

# read training and test data files and create train, validation and test sets and loaders

train_path = database_path + 'train/'
test_path = database_path + 'test/'

train_files = glob(train_path + '*.pkl')
test_files = glob(test_path + '*.pkl')

n_training_max = input_data.get("n_training_max", None)
n_test_max = input_data.get("n_test_max", None)
train_validation_fraction = input_data.get("train_validation_fraction", 0.3)

train_list, valid_list = train_test_split(train_files,
                              test_size=train_validation_fraction)

if n_training_max is None or n_training_max > len(train_list):
    train_dataset = HyperGraphDataSet(files = train_list)
    valid_dataset = HyperGraphDataSet(files = valid_list)
else:
    n_valid_max = int(train_validation_fraction * n_training_max)
    train_dataset = HyperGraphDataSet(files = train_list[:n_training_max])
    valid_dataset = HyperGraphDataSet(files = valid_list[:n_valid_max])

if n_test_max is None or n_test_max > len(test_files):
   test_list = test_files
else: 
   test_list = test_files[:n_test_max]
   
test_dataset = HyperGraphDataSet(files = test_list)
   
# read some parameters for optimisation 

n_epochs = input_data.get("n_epochs", 100)
train_batch_size = input_data.get("training_batch_size", 50)
valid_batch_size = input_data.get("validation_batch_size", 10)
n_checkpoint_freq = input_data.get("n_checkpoint_freq", 10)
learning_rate = input_data.get("learning_rate", 1.0e-2)
momentum = input_data.get("momentum", 0.9)

# create dataloaders for each dataset

train_dl = HyperGraphDataLoader(train_dataset, batch_size = train_batch_size)
valid_dl = HyperGraphDataLoader(valid_dataset, batch_size = valid_batch_size)
test_dl = HyperGraphDataLoader(test_dataset)

# now read model details and create it 

rngs = nnx.Rngs(42)
convolution_layers = input_data["convolution_layers"]
hedge_MLP = input_data["hedge_MLP"]
node_MLP = input_data["node_MLP"]

model = HyperGraphConvolution(
            rngs = rngs,
            conv_layers = convolution_layers,
            node_layers = node_MLP,
            hedge_layers = hedge_MLP
        ) 

"""
print('Energies of train set')
for n, sample in enumerate(train_dl):
    loss = loss_function(model, sample, 'U0')
    print(f'Loss for sample {n}: {loss}')

print('Energies of test set')
for n, sample in enumerate(test_dl):
    loss = loss_function(model, sample, 'U0')
    print(f'Loss for sample {n}: {loss}')

print('Got this far!')
"""

# define and display an optimizer

optimizer = nnx.Optimizer(model, optax.adam(learning_rate, momentum))
metrics = nnx.MultiMetric(
    loss = nnx.metrics.Average('loss')
)

# nnx.display(optimizer)

metrics_history = {
        'train_loss': [],
        'valid_loss': []
        }

for epoch in range(n_epochs):
    
    running_loss = 0.0
    validation_running_loss = 0.0

    for batch in train_dl:
        loss = train_step(model, loss_function, optimizer, metrics, batch)
        running_loss += loss

    for batch in valid_dl:
        loss = eval_step(model, loss_function, metrics, batch)
        validation_running_loss += loss

    print(f'epoch: {epoch}, running_loss: {running_loss}, validation_running_loss: {validation_running_loss}')
    
# let's try now with the test set

model.eval()

for n, batch in enumerate(test_dl):
    prediction = model(batch)
    ground_truth = jnp.array(batch.targets['U0'])
    prediction = prediction[0,0]
    ground_truth = ground_truth[0,0]
    loss = jnp.sqrt((prediction - ground_truth)**2)
    print(f'sample {n}, prediction: {prediction}, ref: {ground_truth}, mse: {loss}')
