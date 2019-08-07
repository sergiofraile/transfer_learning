#!/bin/bash

# Create folders
mkdir models
mkdir datasets

# Inception model
curl https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 --output models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

# Dataset
local_zip="datasets/cats_and_dogs_filtered.zip"
base_dir="datasets/cats_and_dogs_filtered"
curl https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip --output "$local_zip"
python src/generate_dataset.py "$local_zip" "$base_dir"