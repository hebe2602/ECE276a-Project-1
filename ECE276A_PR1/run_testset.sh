#!/bin/bash

# Check if an argument was provided
if [ $# -eq 0 ]
  then
    echo "No argument provided. Please provide a dataset name."
    exit 1
fi

# Set the dataset variable to the provided argument
dataset=$1

# The code from your previous message
echo "Loading $dataset dataset..."
echo "Dataset loaded."
echo "Dataset name: $dataset"

#Run the code 
python3 code/project1_testset.py