#!/bin/bash

resDir="./results/"   # Directory name to save the results
train_x="train_X.csv" # Training dataset filename
train_y="train_Y.csv" # Subtype label filename for training dataset
test_x="test_X.csv"   # Testing dataset filename
test_y="test_Y.csv"   # Subtype label filename for testing dataset
num_gene="100"        # Number of genes
num_cpg="100"         # Number of CpG clusters
num_mirna="100"       # Number of microRNAs

mkdir $resDir
python BRCAnet.py $train_x $train_y $test_x $test_y $num_gene $num_cpg $num_mirna $resDir

