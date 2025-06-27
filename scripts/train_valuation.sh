#!/bin/bash

VALUATION_MODEL_TYPE="mlp"

# logic
#EXTRA_ARGS="--env-max-ep-steps 3000 --extra-env-modifications disable_monkeys --actor-mode logic --reward-fn goal_with_step_penalty --blend-ent-coef 0 --concept-coef 0.01"
# blender
EXTRA_ARGS="--actor-mode hybrid --reward-fn default --reset-logic-actor --learn-logic-actor --reset-blending-weights --learn-blending-weights --concept-coef 0.01"

NR="105_gpt"
CUDA_DEVICES="0"
NUM_ENVS=128
END_SEED=3
START_SEED=0

# LLM-based oracle
ORACLE="fixed-static" # comment line to disable oracle
#ORACLE_PROGRAM="claude_4sonnet_v3_17x17"
ORACLE_PROGRAM="chatgpt_4o_v3_17x17"

# Static oracle
#ORACLE="relative-static"

# left
#STATIC_PREDICATES="right_of_ladder on_ladder close_by_monkey close_by_throwncoconut nothing_around"

# left + right
#STATIC_PREDICATES="on_ladder close_by_monkey close_by_throwncoconut nothing_around"

# left + right + up
#STATIC_PREDICATES="close_by_monkey close_by_throwncoconut nothing_around"

# left + right + up + close_by_*
STATIC_PREDICATES="nothing_around"

VALMODEL_ARGS="--valuation-model.static-predicates ${STATIC_PREDICATES} --valuation-model.discard-missing-objects --valuation-model.use-position-difference"


# Oracle Args
if [[ -v ORACLE ]]; then
  ORACLE_EXTRA_ARGS=""
  if [[ -v ORACLE_PROGRAM ]]; then
      ORACLE_EXTRA_ARGS="--oracle-model.program ${ORACLE_PROGRAM}"
  fi
  ORACLE_ARGS="oracle-model:${ORACLE} --oracle-model.static-predicates ${STATIC_PREDICATES} ${ORACLE_EXTRA_ARGS}"
else
  ORACLE_ARGS=""
fi

# Args
DEFAULT_ARGS="--wandb-entity ${WANDB_TEAM} \
--rules small --env-frameskip 4 \
--learn-logic-critic --reset-logic-critic --randomize-start-position --log-heatmaps-steps 200000 --save-train-data \
--env-name kangaroo --track --num-steps 128 --recover --save-steps 10000 --total-timesteps 20000000 \
--num-envs ${NUM_ENVS} \
${EXTRA_ARGS:-} \
valuation-model:${VALUATION_MODEL_TYPE} ${VALMODEL_ARGS:-} \
${ORACLE_ARGS}"

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
    --restart on-failure \
    --name "${EXPERIMENT_NAME}" \
    -e WANDB_API_KEY="${WANDB_API_KEY}" \
    -e WANDB_TEAM="${WANDB_TEAM}" \
    blendrl:base \
    bash -c "cd /app & python train_valuation.py --exp-name ${EXPERIMENT_NAME} --seed ${SEED} ${DEFAULT_ARGS}" > /dev/null 2>&1 &

done
