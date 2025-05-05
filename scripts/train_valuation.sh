#!/bin/bash

VALUATION_MODEL_TYPE="dlgn"
EXTRA_ARGS="--valuation-model.hidden-sizes 256 256 256 256 256"
#EXTRA_ARGS="--valuation-model.static-predicates on_ladder right_of_ladder on_pl_ladder on_pl_player close_by_fruit close_by_bell close_by_monkey close_by_throwncoconut close_by_fallingcoconut nothing_around same_level_ladder"
NR="004"
CUDA_DEVICES="1"
NUM_ENVS=96
START_SEED=0
END_SEED=3

# Args
DEFAULT_ARGS="--wandb-entity ${WANDB_TEAM} \
--env-name kangaroo --num-steps 128 --track --recover --save-steps 10000 \
--num-envs ${NUM_ENVS} \
valuation-model:${VALUATION_MODEL_TYPE} ${EXTRA_ARGS}"

# Experiment name
BASE_EXPERIMENT_NAME="${VALUATION_MODEL_TYPE}_${NR}"

# Docker GPU flag
if [ "$CUDA_DEVICES" == "all" ]; then
  DOCKER_GPUS="all"
else
  DOCKER_GPUS="\"device=$CUDA_DEVICES\""
fi

# Navigate to the parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# iterate all seeds
for ((SEED=START_SEED; SEED<END_SEED; SEED++)); do

  EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_s${SEED}"

  # Stop and remove container if still running
  docker stop "${EXPERIMENT_NAME}" 2>/dev/null || true
  docker rm "${EXPERIMENT_NAME}" 2>/dev/null || true

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
    bash -c "cd /app & python train_valuation.py --exp-name ${EXPERIMENT_NAME} --seed ${SEED} ${DEFAULT_ARGS}" > /dev/null 2>&1 &

done
