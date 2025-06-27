#!/bin/bash

EXP_NAMES=("mlp_105_cl_s0" "mlp_105_cl_s1" "mlp_105_cl_s2" "mlp_105_gpt_s0" "mlp_105_gpt_s1" "mlp_105_gpt_s2")
#EXP_NAMES=("mlp_091_s0" "mlp_092_s0" "mlp_093_s0")
#PREDICATES="left_of_ladder"
#PREDICATES="left_of_ladder right_of_ladder"
#PREDICATES="left_of_ladder right_of_ladder on_ladder"
PREDICATES="close_by_monkey close_by_throwncoconut"
#POSITIONS="Ladder_1 Ladder_2 Ladder_3"
POSITIONS="5"
CUDA_DEVICES="0"
SKIP=5
PLOTS="logic_critic valuation_overlay oracle_diff"
#PLOTS="logic_critic"

# Docker GPU flag
if [ "$CUDA_DEVICES" == "all" ]; then
  DOCKER_GPUS="all"
else
  DOCKER_GPUS="\"device=$CUDA_DEVICES\""
fi

# Navigate to the parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

for EXP_NAME in "${EXP_NAMES[@]}"; do

  DOCKER_CONTAINER_NAME="plot_predicate_valuation_${EXP_NAME}"

  # Run docker container in background
  nohup docker run \
    -v $PARENT_DIR:/app \
    --gpus ${DOCKER_GPUS} \
    --ipc=host --ulimit stack=67108864 \
    --rm \
    --name "${DOCKER_CONTAINER_NAME}" \
    blendrl:base \
    bash -c "cd /app & python -m plot.plot_predicate_valuation --exp-name ${EXP_NAME} --predicate-name ${PREDICATES} --position ${POSITIONS} --skip ${SKIP} --contour-objects Ladder Platform --plots ${PLOTS} --logic-critic-extra-objects Monkey ThrownCoconut" > /dev/null 2>&1 &

done