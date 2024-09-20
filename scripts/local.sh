#!/bin/bash

# Prompt the user for desired dataset
read -p "Enter the dataset name: " dataset_name

# Navigate to the datasets directory
cd datasets

# Download and extract dataset
scp -i /home/helix/.ssh/rms dgrzinic@gmn.uwo.ca:datasets/${dataset_name}.tar.bz2 .
tar -xjf ${dataset_name}.tar.bz2
rm ${dataset_name}.tar.bz2
