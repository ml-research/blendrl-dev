#!/bin/bash

STATIC_EXPERIMENT="__static"

simulate()
{
  AGENT_PATH="models/${ENV_NAME}_demo"

  # Docker GPU flag
  if [ "$CUDA_DEVICES" == "all" ]; then
    DOCKER_GPUS="all"
  else
    DOCKER_GPUS="\"device=$CUDA_DEVICES\""
  fi

  # Navigate to the parent directory
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PARENT_DIR="$(dirname "$SCRIPT_DIR")"

  # Iterate all experiments
  for EXP_NAME in "${EXP_NAMES[@]}"; do

    if [ "$EXP_NAME" == "$STATIC_EXPERIMENT" ]; then
      EXP_ARGS=""
      EXP_NAME="static_${ENV_NAME}"
    else
      EXP_ARGS="--exp-name ${EXP_NAME}"
    fi

    for ACTOR_MODE in "${ACTOR_MODES[@]}"; do

      DOCKER_CONTAINER_NAME="sim_valuation_${EXP_NAME}_${ACTOR_MODE}"

      echo "Simulating ${EXP_NAME} (${ACTOR_MODE})"

      # Stop and remove container if still running
      docker stop "${DOCKER_CONTAINER_NAME}" 2>/dev/null || true
      docker rm "${DOCKER_CONTAINER_NAME}" 2>/dev/null || true

      # Run docker container in background
      nohup docker run \
        -v $PARENT_DIR:/app \
        --gpus ${DOCKER_GPUS} \
        --ipc=host --ulimit stack=67108864 \
        --rm \
        --name "${DOCKER_CONTAINER_NAME}" \
        blendrl:base \
        bash -c "cd /app & python sim_valuation.py ${EXP_ARGS} --seed ${SEED} --actor-mode ${ACTOR_MODE} --overwrite --num-episodes ${NUM_EPISODES} --num-envs ${NUM_ENVS} --env-name ${ENV_NAME} --agent-path ${AGENT_PATH} ${EXTRA_ARGS:-}" > /dev/null 2>&1 &

    done

  done
}

SEED=0
CUDA_DEVICES="3"
NUM_ENVS=100
NUM_EPISODES=100
EXTRA_ARGS=""

# ======================================================================================================================
# 1st stage; No enemies
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# KANGAROO
# ----------------------------------------------------------------------------------------------------------------------

ENV_NAME="kangaroo"
ACTOR_MODES=("logic")
EXTRA_ARGS="--extra-env-modifications disable_monkeys --env-frameskip 1 --env-max-ep-steps 3000 --reset-logic-actor"

# Baseline BlendRL
EXP_NAMES=("$STATIC_EXPERIMENT")
#simulate

# Kangaroo Ablations (1st stage; Different Concept Coefficients; With Concept Annealing)
EXP_NAMES=("mlp_209_gpt_s0" "mlp_209_gpt_s1" "mlp_209_gpt_s2" "mlp_209_cl_s0" "mlp_209_cl_s1" "mlp_209_cl_s2" "mlp_257_cl_cc0_1_s0" "mlp_257_cl_cc0_1_s1" "mlp_257_cl_cc0_1_s2" "mlp_257_gpt_cc0_1_s0" "mlp_257_gpt_cc0_1_s1" "mlp_257_gpt_cc0_1_s2" "mlp_257_cl_cc0_3_s0" "mlp_257_cl_cc0_3_s1" "mlp_257_cl_cc0_3_s2" "mlp_257_gpt_cc0_3_s0" "mlp_257_gpt_cc0_3_s1" "mlp_257_gpt_cc0_3_s2" "mlp_257_cl_cc1_0_s0" "mlp_257_cl_cc1_0_s1" "mlp_257_cl_cc1_0_s2" "mlp_257_gpt_cc1_0_s0" "mlp_257_gpt_cc1_0_s1" "mlp_257_gpt_cc1_0_s2")
#simulate

# Baseline Logic
EXP_NAMES=("mlp_213_s513" "mlp_213_s514" "mlp_213_s515")
#simulate

# Baseline ChatGPT + Claude
EXTRA_ARGS="--extra-env-modifications disable_monkeys --env-frameskip 1 --env-max-ep-steps 3000 --use-oracle --reset-logic-actor"
EXP_NAMES=("mlp_257_gpt_cc0_3_s1" "mlp_257_cl_cc0_3_s1")
#simulate


# ----------------------------------------------------------------------------------------------------------------------

ACTOR_MODES=("logic")
EXTRA_ARGS="--extra-env-modifications disable_monkeys --env-frameskip 1 --env-max-ep-steps 3000"

# Baseline BlendRL
EXP_NAMES=("$STATIC_EXPERIMENT")
#simulate

# 1st stage
EXP_NAMES=("mlp_257_gpt_cc0_3_s1" "mlp_257_cl_cc0_3_s1")
EXP_NAMES=("mlp_257_gpt_cc0_3_s0" "mlp_257_gpt_cc0_3_s1" "mlp_257_gpt_cc0_3_s2")
#simulate

# Kangaroo Ablations (1st stage; Different Concept Coefficients; With Concept Annealing)
EXP_NAMES=("mlp_209_gpt_s0" "mlp_209_gpt_s1" "mlp_209_gpt_s2" "mlp_209_cl_s0" "mlp_209_cl_s1" "mlp_209_cl_s2" "mlp_257_cl_cc0_1_s0" "mlp_257_cl_cc0_1_s1" "mlp_257_cl_cc0_1_s2" "mlp_257_gpt_cc0_1_s0" "mlp_257_gpt_cc0_1_s1" "mlp_257_gpt_cc0_1_s2" "mlp_257_cl_cc0_3_s0" "mlp_257_cl_cc0_3_s1" "mlp_257_cl_cc0_3_s2" "mlp_257_gpt_cc0_3_s0" "mlp_257_gpt_cc0_3_s1" "mlp_257_gpt_cc0_3_s2" "mlp_257_cl_cc1_0_s0" "mlp_257_cl_cc1_0_s1" "mlp_257_cl_cc1_0_s2" "mlp_257_gpt_cc1_0_s0" "mlp_257_gpt_cc1_0_s1" "mlp_257_gpt_cc1_0_s2")
#simulate

# Kangaroo Ablations (1st stage; Different Concept Coefficients; No Concept Annealing)
EXP_NAMES=("mlp_209_cl_cc0_03_s0" "mlp_209_cl_cc0_03_s1" "mlp_209_cl_cc0_03_s2" "mlp_209_gpt_cc0_03_s0" "mlp_209_gpt_cc0_03_s1" "mlp_209_gpt_cc0_03_s2" "mlp_209_cl_cc0_1_s0" "mlp_209_cl_cc0_1_s1" "mlp_209_cl_cc0_1_s2" "mlp_209_gpt_cc0_1_s0" "mlp_209_gpt_cc0_1_s1" "mlp_209_gpt_cc0_1_s2" "mlp_209_cl_cc0_3_s0" "mlp_209_cl_cc0_3_s1" "mlp_209_cl_cc0_3_s2" "mlp_209_gpt_cc0_3_s0" "mlp_209_gpt_cc0_3_s1" "mlp_209_gpt_cc0_3_s2" "mlp_209_cl_cc1_0_s0" "mlp_209_cl_cc1_0_s1" "mlp_209_cl_cc1_0_s2" "mlp_209_gpt_cc1_0_s0" "mlp_209_gpt_cc1_0_s1" "mlp_209_gpt_cc1_0_s2")
#simulate

# Baseline Neural
ACTOR_MODES=("neural")
EXP_NAMES=("mlp_214_s513" "mlp_214_s514" "mlp_214_s515")
#simulate

# Baseline ChatGPT + Claude
EXTRA_ARGS="--extra-env-modifications disable_monkeys --env-frameskip 1 --env-max-ep-steps 3000 --use-oracle"
ACTOR_MODES=("logic")
EXP_NAMES=("mlp_257_gpt_cc0_3_s1" "mlp_257_cl_cc0_3_s1")
#simulate


# ----------------------------------------------------------------------------------------------------------------------
# SEAQUEST
# ----------------------------------------------------------------------------------------------------------------------

ENV_NAME="seaquest"
ACTOR_MODES=("logic")
EXTRA_ARGS="--extra-env-modifications disable_enemies --env-frameskip 1 --env-max-ep-steps 3000 --reset-logic-actor"

# Baseline BlendRL
EXP_NAMES=("$STATIC_EXPERIMENT")
#simulate

# 1st stage
#EXP_NAMES=("mlp_309_gpt_s0" "mlp_309_gpt_s1" "mlp_309_gpt_s2" "mlp_309_cl_s0" "mlp_309_cl_s1" "mlp_309_cl_s2" "mlp_357_cl_cc0_1_s0" "mlp_357_cl_cc0_1_s1" "mlp_357_cl_cc0_1_s2" "mlp_357_gpt_cc0_1_s0" "mlp_357_gpt_cc0_1_s1" "mlp_357_gpt_cc0_1_s2" "mlp_357_cl_cc0_3_s0" "mlp_357_cl_cc0_3_s1" "mlp_357_cl_cc0_3_s2" "mlp_357_gpt_cc0_3_s0" "mlp_357_gpt_cc0_3_s1" "mlp_357_gpt_cc0_3_s2" "mlp_357_cl_cc1_0_s0" "mlp_357_cl_cc1_0_s1" "mlp_357_cl_cc1_0_s2" "mlp_357_gpt_cc1_0_s0" "mlp_357_gpt_cc1_0_s1" "mlp_357_gpt_cc1_0_s2")
EXP_NAMES=("mlp_309_cl_cc0_03_s0" "mlp_309_cl_cc0_03_s1" "mlp_309_cl_cc0_03_s2" "mlp_309_gpt_cc0_03_s0" "mlp_309_gpt_cc0_03_s1" "mlp_309_gpt_cc0_03_s2" "mlp_309_cl_cc0_1_s0" "mlp_309_cl_cc0_1_s1" "mlp_309_cl_cc0_1_s2" "mlp_309_gpt_cc0_1_s0" "mlp_309_gpt_cc0_1_s1" "mlp_309_gpt_cc0_1_s2" "mlp_309_cl_cc0_3_s0" "mlp_309_cl_cc0_3_s1" "mlp_309_cl_cc0_3_s2" "mlp_309_gpt_cc0_3_s0" "mlp_309_gpt_cc0_3_s1" "mlp_309_gpt_cc0_3_s2" "mlp_309_cl_cc1_0_s0" "mlp_309_cl_cc1_0_s1" "mlp_309_cl_cc1_0_s2" "mlp_309_gpt_cc1_0_s0" "mlp_309_gpt_cc1_0_s1" "mlp_309_gpt_cc1_0_s2")
#EXP_NAMES=("mlp_309_gpt_s1" "mlp_309_cl_s1" "mlp_309_gpt_s2" "mlp_309_cl_s2") # "mlp_309_gpt_s0" "mlp_309_cl_s0"
#simulate

# Baseline Logic
EXP_NAMES=("mlp_313_s1024" "mlp_313_s1025" "mlp_313_s1026")
#simulate

ACTOR_MODES=("logic")
EXTRA_ARGS="--extra-env-modifications disable_enemies --env-frameskip 1 --env-max-ep-steps 3000"

# Baseline BlendRL
EXP_NAMES=("$STATIC_EXPERIMENT")
#simulate

# 1st stage
EXP_NAMES=("mlp_309_gpt_s0" "mlp_309_cl_s0")
#simulate

# Seaquest Ablations (1st stage; Different Concept Coefficients; With Concept Annealing)
EXP_NAMES=("mlp_309_gpt_s0" "mlp_309_gpt_s1" "mlp_309_gpt_s2" "mlp_309_cl_s0" "mlp_309_cl_s1" "mlp_309_cl_s2" "mlp_357_cl_cc0_1_s0" "mlp_357_cl_cc0_1_s1" "mlp_357_cl_cc0_1_s2" "mlp_357_gpt_cc0_1_s0" "mlp_357_gpt_cc0_1_s1" "mlp_357_gpt_cc0_1_s2" "mlp_357_cl_cc0_3_s0" "mlp_357_cl_cc0_3_s1" "mlp_357_cl_cc0_3_s2" "mlp_357_gpt_cc0_3_s0" "mlp_357_gpt_cc0_3_s1" "mlp_357_gpt_cc0_3_s2" "mlp_357_cl_cc1_0_s0" "mlp_357_cl_cc1_0_s1" "mlp_357_cl_cc1_0_s2" "mlp_357_gpt_cc1_0_s0" "mlp_357_gpt_cc1_0_s1" "mlp_357_gpt_cc1_0_s2")
#simulate

# Seaquest Ablations (1st stage; Different Concept Coefficients; No Concept Annealing)
EXP_NAMES=("mlp_309_cl_cc0_03_s0" "mlp_309_cl_cc0_03_s1" "mlp_309_cl_cc0_03_s2" "mlp_309_gpt_cc0_03_s0" "mlp_309_gpt_cc0_03_s1" "mlp_309_gpt_cc0_03_s2" "mlp_309_cl_cc0_1_s0" "mlp_309_cl_cc0_1_s1" "mlp_309_cl_cc0_1_s2" "mlp_309_gpt_cc0_1_s0" "mlp_309_gpt_cc0_1_s1" "mlp_309_gpt_cc0_1_s2" "mlp_309_cl_cc0_3_s0" "mlp_309_cl_cc0_3_s1" "mlp_309_cl_cc0_3_s2" "mlp_309_gpt_cc0_3_s0" "mlp_309_gpt_cc0_3_s1" "mlp_309_gpt_cc0_3_s2" "mlp_309_cl_cc1_0_s0" "mlp_309_cl_cc1_0_s1" "mlp_309_cl_cc1_0_s2" "mlp_309_gpt_cc1_0_s0" "mlp_309_gpt_cc1_0_s1" "mlp_309_gpt_cc1_0_s2")
#simulate

# Baseline Neural
EXP_NAMES=("mlp_314_s1024" "mlp_314_s1025" "mlp_314_s1026")
ACTOR_MODES=("neural")
#simulate

# Baseline ChatGPT + Claude
EXTRA_ARGS="--extra-env-modifications disable_enemies --env-frameskip 1 --env-max-ep-steps 3000 --use-oracle"
ACTOR_MODES=("logic")
EXP_NAMES=("mlp_309_gpt_s0" "mlp_309_cl_s0")
#simulate

# Baseline ChatGPT + Claude (reset clause weights)
EXTRA_ARGS="--extra-env-modifications disable_enemies --env-frameskip 1 --env-max-ep-steps 3000 --use-oracle --reset-logic-actor"
ACTOR_MODES=("logic")
EXP_NAMES=("mlp_309_gpt_s0" "mlp_309_cl_s0")
#simulate


# ----------------------------------------------------------------------------------------------------------------------
# SKIING
# ----------------------------------------------------------------------------------------------------------------------
ENV_NAME="skiing"
ACTOR_MODES=("logic")
EXTRA_ARGS="--env-frameskip 1 --env-max-ep-steps 3000 --reset-logic-actor"

# 1st stage (GRAIL + GPT4o) CC=0.1
EXP_NAMES=("mlp_477_1st_gpt_cc0_1_s0" "mlp_477_1st_gpt_cc0_1_s1" "mlp_477_1st_gpt_cc0_1_s2")
#simulate

# 1st stage (GRAIL + Claude4) CC=0.1
EXP_NAMES=("mlp_477_1st_cl_cc0_1_s0" "mlp_477_1st_cl_cc0_1_s1" "mlp_477_1st_cl_cc0_1_s2")
#simulate

# 1st stage (GRAIL + GPT4o) CC=0.3
EXP_NAMES=("mlp_477_1st_gpt_cc0_3_s0" "mlp_477_1st_gpt_cc0_3_s1" "mlp_477_1st_gpt_cc0_3_s2")
#simulate

# 1st stage (GRAIL + Claude4) CC=0.3
EXP_NAMES=("mlp_477_1st_cl_cc0_3_s0" "mlp_477_1st_cl_cc0_3_s1" "mlp_477_1st_cl_cc0_3_s2")
#simulate

# 1st stage (GRAIL + GPT4o) CC=1.0
EXP_NAMES=("mlp_477_1st_gpt_cc1_0_s0" "mlp_477_1st_gpt_cc1_0_s1" "mlp_477_1st_gpt_cc1_0_s2")
#simulate

# 1st stage (GRAIL + Claude4) CC=1.0
EXP_NAMES=("mlp_477_1st_cl_cc1_0_s0" "mlp_477_1st_cl_cc1_0_s1" "mlp_477_1st_cl_cc1_0_s2")
#simulate

# ----------------------------------
# Baseline BlendRL (w/o proxy)
EXP_NAMES=("mlp_413_oriented_v1_s0" "mlp_413_oriented_v1_s1" "mlp_413_oriented_v1_s2")
#simulate

# ----------------------------------
# Baseline Neural PPO
ACTOR_MODES=("neural")
EXP_NAMES=("mlp_414a_s0" "mlp_414a_s1" "mlp_414a_s2")
simulate

# ----------------------------------
# Baseline Oracles (BlendRL + GPT4o)
EXTRA_ARGS="--env-frameskip 1 --env-max-ep-steps 3000 --use-oracle --reset-logic-actor"
ACTOR_MODES=("logic")
EXP_NAMES=("mlp_477_1st_gpt_cc0_1_s0")
#simulate

# Baseline Oracles (BlendRL + Claude4)
EXTRA_ARGS="--env-frameskip 1 --env-max-ep-steps 3000 --use-oracle  --reset-logic-actor"
ACTOR_MODES=("logic")
EXP_NAMES=("mlp_477_1st_cl_cc0_1_s0")
#simulate

# ======================================================================================================================
# Hybrid; With enemies
# ======================================================================================================================

# ----------------------------------------------------------------------------------------------------------------------
# KANGAROO
# ----------------------------------------------------------------------------------------------------------------------

ENV_NAME="kangaroo"
ACTOR_MODES=("hybrid")
EXTRA_ARGS="--env-frameskip 1"

# Baseline: BlendRL
EXP_NAMES=("$STATIC_EXPERIMENT")
#simulate

# 2nd stage; c_CA=0.3; CA-annealing
EXP_NAMES=("mlp_216_gpt_def_s513" "mlp_216_gpt_def_s514" "mlp_216_gpt_def_s515" "mlp_216_cl_def_s513" "mlp_216_cl_def_s514" "mlp_216_cl_def_s515")
#simulate

# 2nd stage; c_CA=0.3; CA-annealing; goal only
EXP_NAMES=("mlp_216_gpt_go_s513" "mlp_216_gpt_go_s514" "mlp_216_cl_go_s513" "mlp_216_cl_go_s514")
#simulate

# 2nd stage
EXP_NAMES=("mlp_210_gpt_s513" "mlp_210_gpt_s514" "mlp_210_gpt_s515" "mlp_210_cl_s513" "mlp_210_cl_s514" "mlp_210_cl_s515")
#simulate

# Ablation: 2nd stage, reward functions
#EXP_NAMES=("mlp_251_go_s515" "mlp_251_wl_s515" "mlp_251_wdp_s515" "mlp_251_wladp_s515")
EXP_NAMES=("mlp_251_defwdp_s515")
#simulate

# Ablation: 2nd stage, concept coefficients
EXP_NAMES=("mlp_252_cl_cc0_1_s515" "mlp_252_cl_cc0_3_s515" "mlp_252_cl_cc1_0_s515" "mlp_252_cl_cc3_0_s515")
#simulate

# Baseline: Neural
ACTOR_MODES=("neural")
EXP_NAMES=("mlp_212_s513" "mlp_212_s514" "mlp_212_s515")
#simulate

# ----------------------------------------------------------------------------------------------------------------------
# SEAQUEST
# ----------------------------------------------------------------------------------------------------------------------

ENV_NAME="seaquest"
ACTOR_MODES=("hybrid")
EXTRA_ARGS="--env-frameskip 1"

# Baseline: BlendRL
EXP_NAMES=("$STATIC_EXPERIMENT")
#simulate

# 2nd stage; c_CA=0.3; CA-annealing
EXP_NAMES=("mlp_316_gpt_def_s1024" "mlp_316_gpt_def_s1025" "mlp_316_gpt_def_s1026" "mlp_316_cl_def_s1024" "mlp_316_cl_def_s1025" "mlp_316_cl_def_s1026")
#simulate

# 2nd stage; c_CA=0.3; CA-annealing; goal only
EXP_NAMES=("mlp_316_gpt_go_s1024" "mlp_316_gpt_go_s1025" "mlp_316_cl_go_s1024" "mlp_316_cl_go_s1025")
#simulate

# Baseline: Neural
ACTOR_MODES=("neural")
EXP_NAMES=("mlp_312_s1024" "mlp_312_s1025" "mlp_312_s1026")
#simulate