
from datetime import datetime
import os
import pathlib
import pdb
import re
import sys

import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import yaml

from generate_hypergraphs import generate_hypergraphs
from logging_utilities import write_node_features
from set_up_hypergraphs import set_up_hypergraphs

"""
A script to construct hyper-graph representations of atomic systems.

To execute: python graph_maker.py input-file

where input-file is a yaml file specifying how the graphs are to be
constructed and from which source file(s). The graphs will be stored
in the indicated directory tree in the form of json files. 

A sample input file is graph_maker_sample.yml

"""

input_file = sys.argv[1]  # input_file is a yaml compliant file

with open( input_file, 'r' ) as input_stream:
    input_data = yaml.load(input_stream, Loader=yaml.Loader)

debug_mode = input_data.get("debug", False)

if debug_mode:
    pdb.set_trace()

source_directory = input_data.get("DataBaseDirectory",
                                  "~/Work/Databases/QM9DataBase/")
target_directory = input_data.get("TargetDirectory", "./QM9GraphDataBase/")

# check if the source directory already contains train and test subdirectories
# a validate data set will be obtained splitting from train data set

split_database = False

train = os.path.join(source_directory + "/train/")
test = os.path.join(source_directory + "/test/")

if os.path.exists(train) and os.path.isdir(train) and \
   os.path.exists(test) and os.path.isdir(test):

   split_database = True  # database is already split, so no need to split it

if not split_database:

   data_base = pathlib.Path(source_directory)
   test_split_fraction = input_data.get("SplitFractions", 0.3)

# now, if necessary, create the target directory and its subdirectories

train_path = pathlib.Path(train)
test_path = pathlib.Path(test)

target_path = pathlib.Path(target_directory)
target_train_path = pathlib.Path(target_directory + "/train" )
target_test_path = pathlib.Path(target_directory + "/test" )

if not target_path.exists():

   target_path.mkdir()
   target_train_path.mkdir()
   target_test_path.mkdir()

else:

   if not target_train_path.exists(): target_train_path.mkdir()
   if not target_test_path.exists(): target_test_path.mkdir()

# file extensions

input_file_ext = input_data.get("InputFileExtension", '.xyz')
output_file_ext = input_data.get("OutputFileExtension", '.pkl')

# generate respective graph databases

pattern = '*' + input_file_ext

if split_database:

   train_files = train_path.glob(pattern)
   test_files = test_path.glob(pattern)

else:

   files_list = list( data_base.glob(pattern) )

   # generator = torch.Generator().manual_seed(42)

   train_list, test_list = train_test_split( files_list,
                    test_size = test_split_fraction)

   # convert these lists to generators

   train_files = (file for file in train_list)
   test_files = (file for file in test_list)

# now find out what class of graph (standard or heterograph) and
# what type of graphs (dependent on the class) we need to construct

# graphClass = input_data.get("graphClass", "standard")

graph_type = input_data.get("graphType", "Covalent")  # default is Covalent

# now specify the graph construction strategy

n_max_neighbours = input_data.get("nMaxNeighbours", 6)
use_covalent_radii = input_data.get("useCovalentRadii", False)
node_features = input_data.get("nodeFeatures", [])
n_hedge_features = input_data.get("nEdgeFeatures", 10)
species = input_data.get("species", ["H", "C", "N", "O", "F"])
pooling = input_data.get("pooling", "add")

# FOLLOWING COMMENTS NEED TO BE UPDATED!!!!!
#
# note that nNodeFeatures >= len(nodeFeatures); node features are 
# of three types: 
# 
# 1) two one-hot encoded features, one for the species period (7 values)
#    and one for the species group (18 values), i.e. a 25-long 
#    one-hot encoded vector where only two values are non-zero
#
# 2) a number (possibly zero) of species properties given as Mendeleev
#    recognized commands, e.g. "nodeFeatures = ["atomic_number", "covalent_radius"]"
#    etc.
# 
# 3) "bond-angle geometric features", i.e., optionally it is possible to 
#    append to the previous two a vector of bond-angle features; this is 
#    a feature vector of length bond_angle.n_features() which is a histogram
#    of bond-angle values centred on the particular node. Contrary to the previous
#    two types, which are common to all atoms of the same species, the geometric
#    features depend on the particular environment of each atom, and thus
#    encode the local environment right from the beginning. 

# read and set-up edge, bond-angle and dihedral-angle features

# features_dict = set_up_features(input_data)

transformData = input_data.get("transformData", False)
# transform = SetUpDataTransform( transformData, directories )
transform = None

Graphs = set_up_hypergraphs(
    graph_type = graph_type,
    species = species, 
    node_feature_list = node_features,
    n_hedge_features = n_hedge_features,
    n_max_neighbours = n_max_neighbours,
    pooling = pooling
)

# if re.match("^stand", graphClass):

n_node_features = Graphs.n_node_features

write_node_features(node_features)

descriptionText = input_data.get("descriptionText", " ")

# now proceed to generate the graphs for training, validation and test datasets

make_graphs = input_data.get("generate_graphs", True)

if make_graphs:
   generate_hypergraphs( Graphs, train_files,
                         target_train_path, output_file_ext )
   generate_hypergraphs( Graphs, test_files, 
                         target_test_path, output_file_ext )

# save the description text as a new key to the yaml file

# finally, we will write a yaml file containing a description of the 
# graph construction; this will be later read and employed by the 
# graph fitting process

description_file = target_directory + '/' + 'graph_description.yml'

# we append to the input data information on the total number of node features

input_data['nNodeFeatures'] = n_node_features
input_data['nEdgeFeatures'] = (n_hedge_features+1) # +1 for spin

with open( description_file, 'w' ) as description:
    yaml.dump(input_data, description)

