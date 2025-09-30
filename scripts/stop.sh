#!/bin/bash

NUM_SEEDS=3

# Stop and remove container if still running
for EXP_NAME in "$@"
do
    for ((SEED=0; SEED<NUM_SEEDS; SEED++)); do
      docker stop "${EXP_NAME}_s${SEED}" 2>/dev/null || true
      docker rm "${EXP_NAME}_s${SEED}" 2>/dev/null || true
    done
done
