#!/bin/bash

EXP_NAME="sgn_018_s0"
PREDICATES="left_of_ladder"
#PREDICATES="left_of_ladder right_of_ladder"
#PREDICATES="left_of_ladder right_of_ladder on_ladder"
POSITIONS="Ladder_1 Ladder_2 Ladder_3"
CUDA_DEVICES="3"
SKIP=5

DOCKER_UID=$(uuidgen | cut -c1-8)
DOCKER_CONTAINER_NAME="plot_predicate_valuation_${DOCKER_UID}"

# Docker GPU flag
if [ "$CUDA_DEVICES" == "all" ]; then
  DOCKER_GPUS="all"
else
  DOCKER_GPUS="\"device=$CUDA_DEVICES\""
fi

# Navigate to the parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Run docker container in background
nohup docker run \
  -v $PARENT_DIR:/app \
  --gpus ${DOCKER_GPUS} \
  --ipc=host --ulimit stack=67108864 \
  --rm \
  --name "${DOCKER_CONTAINER_NAME}" \
  blendrl:base \
  bash -c "cd /app & python -m plot.plot_predicate_valuation --exp-name ${EXP_NAME} --predicate-name ${PREDICATES} --position ${POSITIONS} --skip ${SKIP} --overlay-positions --obj-overlays Ladder Platform --plot-logic-critic" > /dev/null 2>&1 &
