#!/bin/bash

architectures=("deformable" "rigid")
subsampling_values=(0.20 0.15 0.1)

for arch in "${architectures[@]}"}; do
    for subsample in "${subsampling_values[@]}"; do
        python runmodels.py $arch $subsample
    done
done