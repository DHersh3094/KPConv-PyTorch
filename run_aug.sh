#!/bin/bash

rotations=(0 3 6)
decimation_runs=(0 3 6)
jitter_amounts=(0 0.01 0.02)

for rot in "${rotations[@]}"; do
    for decim in "${decimation_runs[@]}"; do
        for jit in "${jitter_amounts[@]}"; do
            echo "Running with rotation=$rot, decimation_runs=$decim, jitter=$jit"
            
            # Inputs: 1: rotation, 2: decimation_runs 3: jitter
            # rotation_input = sys.argv[1]
            # decimation_runs_input = sys.argv[2]
            # jitter_amount_input = sys.argv[3] 
            python runmodels.py $rot $decim $jit
        done
    done
done