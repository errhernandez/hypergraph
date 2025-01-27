
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
from evaluate_model import evaluate_energy
from hypergraph_batch import hypergraph_batch
from hypergraph_model import HyperGraphConvolution 
from hypergraph_dataset import HyperGraphDataSet
from hypergraph_dataloader import HyperGraphDataLoader
from loss_function import loss_function
from run_banner import run_banner
from training import train_model

"""

This script reads in a previously fitted hypergraph model and uses it to
to calculate the energy and the forces on a set of trial molecules.

To execute: python check_forces.py input-file

where input-file is a yaml file; the same input file can be used
as was used to fit the model in the first place, making sure that
now the model parameters are read in instead of re-fitted. 

"""

input_file = sys.argv[1]   # input_file is ayaml compliant file

with open(input_file, 'r') as input_stream:
    input_data = yaml.load(input_stream, Loader=yaml.Loader)

debug = input_data.get("debug", False)

if debug:
    faulthandler.enable()
    pdb.set_trace()

time_string = datetime.now().strftime("%d%m%Y-%H%M%S")

# create a log file for the run

log_dir = input_data.get("log_dir", './runs/')
log_file_str = input_data.get("log_file", 'test_run')
log_file = log_dir + log_file_str + '_' + time_string + '.log'
checkpoint_dir = input_data.get("checkpoint_dir", './checkpoints/')
checkpoint_file = checkpoint_dir + 'checkpoint_' + time_string + '.js'

if not os.path.exists(log_dir):  # check if we need to create the log dir
    log_path = pathlib.Path(log_dir)
    log_path.mkdir()

# create an instance of SummaryWriter for logging purposes

writer = SummaryWriter(log_file)

banner = run_banner(time_string)
print(banner) # print it also to standard output
banner = markdown.markdown(banner)

writer.add_text("Banner", banner)

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
n_start = input_data.get("n_start", 0)
n_print = input_data.get("n_print", 1) 
train_batch_size = input_data.get("training_batch_size", 50)
valid_batch_size = input_data.get("validation_batch_size", 10)
n_checkpoint_freq = input_data.get("n_checkpoint_freq", 10)

# following parameters pertain to the Optax "reduce_on_plateau" learning-rate scheduler 
learning_rate = input_data.get("learning_rate", 1.0e-2)
lr_reduction_factor = input_data.get("lr_reduction_factor", 1.0e-1)
lr_patience = input_data.get("lr_patience", 10)
lr_rtol = input_data.get("lr_rtol", 1.0e-4)
lr_atol = input_data.get("lr_atol", 1.0e-5)
momentum = input_data.get("momentum", 0.9)

# create dataloaders for each dataset

train_dl = HyperGraphDataLoader(train_dataset, batch_size = train_batch_size)
valid_dl = HyperGraphDataLoader(valid_dataset, batch_size = valid_batch_size)
test_dl = HyperGraphDataLoader(test_dataset)

# now read model details and create it 

key = jax.random.PRNGKey(42)
convolution_layers = input_data["convolution_layers"]

# the first convolution layer has its input determined by the node and hedge 
# feature sizes encoded in the database, so for this layer, we must take these if 
# what the user specifies in the input file is different

convolution_layers[0]['n_hedge_in'] = database_data['nEdgeFeatures']
convolution_layers[0]['n_node_in'] = database_data['nNodeFeatures']

hedge_MLP = input_data["hedge_MLP"]
node_MLP = input_data["node_MLP"]

model = HyperGraphConvolution(
            key = key,
            conv_layers = convolution_layers,
            node_layers = node_MLP,
            hedge_layers = hedge_MLP
        ) 

hyperparams = {'conv_layers': convolution_layers,
               'node_layers': node_MLP,
               'hedge_layers': hedge_MLP}

# check if we are loading model parameters from a previous checkpoint

load_model = input_data.get("load_model", False)

if load_model:

   restart_file = input_data.get("load_model_file", None)

   if restart_file is None:

      print(f'For a re-start optimisation job you must provide a state file!')
      print(f'Use command load_model_file to do so')
      sys.exit()

   elif not os.path.exists(restart_file):

      print(f'Re-start file does not exist!')
      sys.exit()

   else: # file specified and exists, so call model_restore function

      model = checkpoint_load(restart_file)

# final_model.eval()

print("         TEST SAMPLE ENERGIES             ")
print("------------------------------------------")

ref = []
prd = []
lss = []
abslss = []

derivative_nodes = jax.grad(evaluate_energy, argnums=2)
derivative_hedges = jax.grad(evaluate_energy, argnums=3)

params, static = eqx.partition(model, eqx.is_array)

for n, batch in enumerate(test_dl):

    prediction = model(batch.node_features, batch.hedge_features, batch)
    ground_truth = jnp.array(batch.targets['U0'])
    prediction = prediction[0,0]
    ground_truth = ground_truth[0,0]
    loss = jnp.sqrt((prediction - ground_truth)**2)

    energy = evaluate_energy(
               params,
               static,
               batch.node_features,
               batch.hedge_features,
               batch
    )

    print(f'sample {n}, prediction: {prediction}, energy: {energy}, ref: {ground_truth}, mse: {loss}')

    grad_hedges = derivative_hedges(params, static, batch.node_features, batch.hedge_features, batch)

    ref.append(ground_truth)
    prd.append(prediction)
    lss.append(prediction - ground_truth)
    abslss.append(loss)

reference = np.array(ref)
predicted = np.array(prd)
difference = np.array(lss)
sqrtloss = np.array(abslss)

"""
if writer is not None:

    e_max = np.max(reference)
    e_min = np.min(reference)

    x = np.linspace(e_min, e_max, 100)

    figPredVsExN = plt.figure()
    plt.plot(reference, predicted, "bo", label="model predictions")
    plt.plot(x, x, "r-", label="exact")
    plt.legend()

    writer.add_figure("Prediction vs. exact ", figPredVsExN, n_epochs)
    writer.add_histogram(
        "Distribution of errors normalised data (prediction - exact)", difference
    )

writer.close()
"""
