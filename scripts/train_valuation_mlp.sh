#!/bin/bash

DEFAULT_EXPERIMENT_NAME="mlp_007"
CUDA_DEVICES="3"
NUM_ENVS=96

# Experiment name
EXPERIMENT_NAME=${1:-${DEFAULT_EXPERIMENT_NAME}}

# Docker container name
DOCKER_CONTAINER_NAME=${EXPERIMENT_NAME}

# Docker GPU flag
if [ "$CUDA_DEVICES" == "all" ]; then
  DOCKER_GPUS="all"
else
  DOCKER_GPUS="\"device=$CUDA_DEVICES\""
fi

# Navigate to the parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Stop and remove container if still running
docker stop "${DOCKER_CONTAINER_NAME}" 2>/dev/null || true
docker rm "${DOCKER_CONTAINER_NAME}" 2>/dev/null || true

# Run docker container in background
nohup docker run \
  -v $PARENT_DIR:/app \
  --gpus ${DOCKER_GPUS} \
  --ipc=host --ulimit stack=67108864 \
  --restart unless-stopped \
  --name "${EXPERIMENT_NAME}" \
  -e WANDB_API_KEY="${WANDB_API_KEY}" \
  -e WANDB_TEAM="${WANDB_TEAM}" \
  blendrl:base \
  bash -c "cd /app & python train_valuation.py --wandb-entity ${WANDB_TEAM} --env-name kangaroo --num-steps 128 --num-envs ${NUM_ENVS} --track --recover --save-steps 10000 --exp-name ${EXPERIMENT_NAME}" &
