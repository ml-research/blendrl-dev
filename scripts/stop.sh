#!/bin/bash

NUM_SEEDS=$2

# Stop and remove container if still running
for ((SEED=0; SEED<NUM_SEEDS; SEED++)); do
  docker stop "$1_s$SEED" 2>/dev/null || true
  docker rm "$1_s$SEED" 2>/dev/null || true
done