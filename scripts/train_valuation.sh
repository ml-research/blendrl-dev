#!/bin/bash

VALUATION_MODEL_TYPE="mlp"
CUDA_DEVICES="5"
NUM_ENVS=128
END_SEED=3
START_SEED=0
HAS_DEMO="1"
RULES="small"

run_experiment()
{
  # Oracle Args
  if [[ -v ORACLE ]]; then
    ORACLE_EXTRA_ARGS=""
    if [[ -v ORACLE_PROGRAM ]]; then
        ORACLE_EXTRA_ARGS="--oracle-model.program ${ORACLE_PROGRAM}"
        EXTRA_ARGS="${EXTRA_ARGS:-} --oracle-num-samples $((ORACLE_RESOLUTION * ORACLE_RESOLUTION))"
    fi
    ORACLE_ARGS="oracle-model:${ORACLE} --oracle-model.static-predicates ${STATIC_PREDICATES} ${ORACLE_EXTRA_ARGS}"
  else
    ORACLE_ARGS=""
  fi

  VALMODEL_ARGS="--valuation-model.static-predicates ${STATIC_PREDICATES} --valuation-model.discard-missing-objects --valuation-model.use-position-difference --valuation-model.clip-value 0.01"

  # Args
  DEFAULT_ARGS=""
  if [ "${HAS_DEMO}" == "1" ]; then
    DEFAULT_ARGS="--agent-path models/${ENV_NAME}_demo"
  fi

  DEFAULT_ARGS="${DEFAULT_ARGS} \
  --wandb-entity ${WANDB_TEAM} \
  --rules ${RULES} \
  --randomize-start-position --log-heatmaps-steps 200000 --save-train-data \
  --env-name ${ENV_NAME} --track --num-steps 128 --recover --save-steps 10000 \
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
    echo "Starting ${EXPERIMENT_NAME}"

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
      bash -c "cd /app & python3 train_valuation.py --exp-name ${EXPERIMENT_NAME} --seed ${SEED} ${DEFAULT_ARGS}" > /dev/null 2>&1 &

  done
}

# ======================================================================================================================
# 1st stage
# BASELINES
# ======================================================================================================================
# Logic
# Neural Critic / Actor; Blender  = not used
# Logic Critic / Actor            = init. randomly; finetune
# Valuation Functions             = init. randomly; finetune
# No Oracle
unset ORACLE

# ----------------------------------------------------------------------------------------------------------------------
# Kangaroo
# ----------------------------------------------------------------------------------------------------------------------
EXTRA_ARGS="--reset-neural-component --reset-blending-weights --reset-logic-critic --learn-logic-critic --reset-logic-actor --learn-logic-actor --total-timesteps 10000000 --actor-mode logic --extra-env-modifications disable_monkeys --reward-fn goal_only --env-frameskip 4 --env-max-ep-steps 3000"
ENV_NAME="kangaroo"
NR="213"
STATIC_PREDICATES="nothing_around"
END_SEED=516
START_SEED=513
#run_experiment

# ----------------------------------------------------------------------------------------------------------------------
# Seaquest
# ----------------------------------------------------------------------------------------------------------------------
EXTRA_ARGS="--reset-neural-component --reset-blending-weights --reset-logic-critic --learn-logic-critic --reset-logic-actor --learn-logic-actor --total-timesteps 10000000 --actor-mode logic --extra-env-modifications disable_enemies --reward-fn goal_only --env-frameskip 4 --env-max-ep-steps 3000"
ENV_NAME="seaquest"
NR="313"
STATIC_PREDICATES="visible_diver full_divers not_full_divers oxygen_low"
END_SEED=1027
START_SEED=1024
#run_experiment

# ----------------------------------------------------------------------------------------------------------------------
# Skiing
# ----------------------------------------------------------------------------------------------------------------------
BASE_ARGS="--anneal-concept-coef --reset-neural-component --reset-blending-weights --reset-logic-critic --learn-logic-critic --reset-logic-actor --learn-logic-actor --total-timesteps 10000000 --actor-mode logic --reward-fn goal_only --env-frameskip 1 --env-max-ep-steps 3000"
ENV_NAME="skiing"
STATIC_PREDICATES="true left_oriented right_oriented straight_oriented"
HAS_DEMO="0"
END_SEED=3
START_SEED=0

# Rules = small
NR="413_small"
#run_experiment

# Rules = default
RULES="default"
NR="413_default"
#run_experiment

# Rules = oriented_v1
RULES="oriented_v1"
NR="413_oriented_v1"
#run_experiment

RULES="small"
HAS_DEMO="1"


# ======================================================================================================================
# Neural PPO (without enemies)
# Neural Critic / Actor           = init. randomly; finetune
# Blender; Logic Critic / Actor   = not used
# Valuation Functions             = not used
# No enemies
unset ORACLE

# ----------------------------------------------------------------------------------------------------------------------
# Kangaroo
# ----------------------------------------------------------------------------------------------------------------------
EXTRA_ARGS="--reset-neural-component --learn-neural-component --reset-blending-weights --reset-logic-critic --reset-logic-actor --total-timesteps 10000000 --actor-mode neural --extra-env-modifications disable_monkeys --reward-fn goal_only --env-frameskip 4 --env-max-ep-steps 3000"
ENV_NAME="kangaroo"
NR="214"
STATIC_PREDICATES="nothing_around"
END_SEED=516
START_SEED=513
#run_experiment

# ----------------------------------------------------------------------------------------------------------------------
# Seaquest
# ----------------------------------------------------------------------------------------------------------------------
EXTRA_ARGS="--reset-neural-component --learn-neural-component --reset-blending-weights --reset-logic-critic --reset-logic-actor --total-timesteps 10000000 --actor-mode neural --extra-env-modifications disable_enemies --reward-fn goal_only --env-frameskip 4 --env-max-ep-steps 3000"
ENV_NAME="seaquest"
NR="314"
STATIC_PREDICATES="visible_diver full_divers not_full_divers oxygen_low"
END_SEED=1027
START_SEED=1024
#run_experiment

# ----------------------------------------------------------------------------------------------------------------------
# Skiing
# ----------------------------------------------------------------------------------------------------------------------
EXTRA_ARGS="--reset-neural-component --learn-neural-component --reset-blending-weights --reset-logic-critic --reset-logic-actor --total-timesteps 10000000 --actor-mode neural --reward-fn goal_only --env-frameskip 1 --env-max-ep-steps 3000"
ENV_NAME="skiing"
NR="414a"
STATIC_PREDICATES="true left_oriented right_oriented straight_oriented"
HAS_DEMO="0"
END_SEED=3
START_SEED=0
#run_experiment
HAS_DEMO="1"
# ======================================================================================================================


# ======================================================================================================================
# 2nd stage
# BASELINES
# ======================================================================================================================
# Neural PPO
# Neural Critic / Actor           = init. randomly; finetune
# Blender; Logic Critic / Actor   = not used
# Valuation Functions             = not used
EXTRA_ARGS="--reset-neural-component --learn-neural-component --reset-blending-weights --reset-logic-critic --reset-logic-actor --total-timesteps 60000000 --actor-mode neural --reward-fn default --env-frameskip 1"

# ----------------------------------------------------------------------------------------------------------------------
# Kangaroo
# ----------------------------------------------------------------------------------------------------------------------
ENV_NAME="kangaroo"
NR="212"
STATIC_PREDICATES="nothing_around"
END_SEED=516
START_SEED=513
#run_experiment

# ----------------------------------------------------------------------------------------------------------------------
# Seaquest
# ----------------------------------------------------------------------------------------------------------------------
ENV_NAME="seaquest"
NR="312"
STATIC_PREDICATES="visible_diver full_divers not_full_divers oxygen_low"
END_SEED=1027
START_SEED=1024
#run_experiment

# ======================================================================================================================
# 2nd stage; Without oracle; Different Reward Functions (Baseline)

# ----------------------------------------------------------------------------------------------------------------------
# Kangaroo
# ----------------------------------------------------------------------------------------------------------------------
BASE_ARGS="--reset-neural-component --learn-neural-component --reset-blending-weights --learn-blending-weights --logic-critic-path out_val/runs/mlp_257_gpt_cc0_3_s1 --logic-actor-path out_val/runs/mlp_257_gpt_cc0_3_s1 --valuation-model-path out_val/runs/mlp_257_gpt_cc0_3_s1 --freeze-valuation-model --total-timesteps 60000000 --actor-mode hybrid --env-frameskip 1"

unset ORACLE
ENV_NAME="kangaroo"
STATIC_PREDICATES="nothing_around"
END_SEED=514
START_SEED=513

EXTRA_ARGS="--reward-fn default ${BASE_ARGS}"
NR="251_def"
#run_experiment

EXTRA_ARGS="--reward-fn goal_only ${BASE_ARGS}"
NR="251_go"
#run_experiment

EXTRA_ARGS="--reward-fn goal_with_level ${BASE_ARGS}"
NR="251_wl"
#run_experiment

EXTRA_ARGS="--reward-fn goal_with_death_penalty ${BASE_ARGS}"
NR="251_wdp"
#run_experiment

EXTRA_ARGS="--reward-fn goal_with_level_and_death_penalty ${BASE_ARGS}"
NR="251_wladp"
#run_experiment

EXTRA_ARGS="--reward-fn default_with_death_penalty ${BASE_ARGS}"
NR="251_defwdp"
#run_experiment

# ----------------------------------------------------------------------------------------------------------------------
# Seaquest
# ----------------------------------------------------------------------------------------------------------------------
BASE_ARGS="--reset-neural-component --learn-neural-component --reset-blending-weights --learn-blending-weights --logic-critic-path out_val/runs/mlp_309_cl_s0 --logic-actor-path out_val/runs/mlp_309_cl_s0 --valuation-model-path out_val/runs/mlp_309_cl_s0 --freeze-valuation-model --total-timesteps 60000000 --actor-mode hybrid --env-frameskip 1"

unset ORACLE
ENV_NAME="seaquest"
STATIC_PREDICATES="visible_diver full_divers not_full_divers oxygen_low"
END_SEED=1025
START_SEED=1024

EXTRA_ARGS="--reward-fn default ${BASE_ARGS}"
NR="351_def"
#run_experiment
# ======================================================================================================================


# ======================================================================================================================
# 1st stage
# ======================================================================================================================
# Logic
# Neural Critic / Actor; Blender  = disabled
# Logic Critic / Actor            = init. randomly; finetune
# Valuation Functions             = init. randomly; finetune
# Oracle                          = enabled
END_SEED=3
START_SEED=0

# ======================================================================================================================
# 1st stage; Different Concept Coefficients; With Concept Annealing

# ----------------------------------------------------------------------------------------------------------------------
# Kangaroo
# ----------------------------------------------------------------------------------------------------------------------
BASE_ARGS="--anneal-concept-coef --reset-neural-component --reset-blending-weights --reset-logic-critic --learn-logic-critic --reset-logic-actor --learn-logic-actor --total-timesteps 10000000 --actor-mode logic --extra-env-modifications disable_monkeys --reward-fn goal_only --env-frameskip 4 --env-max-ep-steps 3000"
ORACLE="fixed-static"
ORACLE_RESOLUTION=49
ENV_NAME="kangaroo"
STATIC_PREDICATES="close_by_monkey close_by_throwncoconut nothing_around"

EXTRA_ARGS="--concept-coef 0.1 ${BASE_ARGS}"

NR="257_gpt_cc0_1"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="257_cl_cc0_1"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 0.3 ${BASE_ARGS}"

NR="257_gpt_cc0_3"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="257_cl_cc0_3"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 1.0 ${BASE_ARGS}"

NR="257_gpt_cc1_0"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="257_cl_cc1_0"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment


# ----------------------------------------------------------------------------------------------------------------------
# Seaquest
# ----------------------------------------------------------------------------------------------------------------------
BASE_ARGS="--anneal-concept-coef --reset-neural-component --reset-blending-weights --reset-logic-critic --learn-logic-critic --reset-logic-actor --learn-logic-actor --total-timesteps 10000000 --actor-mode logic --extra-env-modifications disable_enemies --reward-fn goal_only --env-frameskip 4 --env-max-ep-steps 3000"
ORACLE="fixed-static"
ORACLE_RESOLUTION=49
ENV_NAME="seaquest"
STATIC_PREDICATES="visible_diver full_divers not_full_divers oxygen_low close_by_missile close_by_enemy"

EXTRA_ARGS="--concept-coef 0.1 ${BASE_ARGS}"

NR="357_gpt_cc0_1"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="357_cl_cc0_1"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 0.3 ${BASE_ARGS}"

NR="357_gpt_cc0_3"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="357_cl_cc0_3"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 1.0 ${BASE_ARGS}"

NR="357_gpt_cc1_0"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="357_cl_cc1_0"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

# ----------------------------------------------------------------------------------------------------------------------
# Skiing (Rules small)
# ----------------------------------------------------------------------------------------------------------------------
BASE_ARGS="--anneal-concept-coef --reset-neural-component --reset-blending-weights --reset-logic-critic --learn-logic-critic --reset-logic-actor --learn-logic-actor --total-timesteps 10000000 --actor-mode logic --reward-fn goal_only --env-frameskip 1 --env-max-ep-steps 3000"
ORACLE="fixed-static"
ORACLE_RESOLUTION=49
ENV_NAME="skiing"
STATIC_PREDICATES="true"
HAS_DEMO="0"

EXTRA_ARGS="--concept-coef 0.1 ${BASE_ARGS}"

NR="457_1st_gpt_cc0_1"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="457_1st_cl_cc0_1"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 0.3 ${BASE_ARGS}"

NR="457_1st_gpt_cc0_3"
ORACLE_PROGRAM="chatgpt_2025_12_07_v7"
#run_experiment

NR="457_1st_cl_cc0_3"
ORACLE_PROGRAM="claude_4.5sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 1.0 ${BASE_ARGS}"

NR="457_1st_gpt_cc1_0"
ORACLE_PROGRAM="chatgpt_2025_12_07_v7"
#run_experiment

NR="457_1st_cl_cc1_0"
ORACLE_PROGRAM="claude_4.5sonnet_v7"
#run_experiment

HAS_DEMO="1"

# ----------------------------------------------------------------------------------------------------------------------
# Skiing (Rules default)
# ----------------------------------------------------------------------------------------------------------------------
BASE_ARGS="--anneal-concept-coef --reset-neural-component --reset-blending-weights --reset-logic-critic --learn-logic-critic --reset-logic-actor --learn-logic-actor --total-timesteps 10000000 --actor-mode logic --reward-fn goal_only --env-frameskip 1 --env-max-ep-steps 3000"
ORACLE="fixed-static"
ORACLE_RESOLUTION=49
ENV_NAME="skiing"
STATIC_PREDICATES="true"
HAS_DEMO="0"
RULES="default"

EXTRA_ARGS="--concept-coef 0.1 ${BASE_ARGS}"

NR="467_1st_gpt_cc0_1"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="467_1st_cl_cc0_1"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 0.3 ${BASE_ARGS}"

NR="467_1st_gpt_cc0_3"
ORACLE_PROGRAM="chatgpt_2025_12_07_v7"
#run_experiment

NR="467_1st_cl_cc0_3"
ORACLE_PROGRAM="claude_4.5sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 1.0 ${BASE_ARGS}"

NR="467_1st_gpt_cc1_0"
ORACLE_PROGRAM="chatgpt_2025_12_07_v7"
#run_experiment

NR="467_1st_cl_cc1_0"
ORACLE_PROGRAM="claude_4.5sonnet_v7"
#run_experiment

RULES="small"
HAS_DEMO="1"

# ----------------------------------------------------------------------------------------------------------------------
# Skiing (Rules oriented_v1)
# ----------------------------------------------------------------------------------------------------------------------
BASE_ARGS="--anneal-concept-coef --reset-neural-component --reset-blending-weights --reset-logic-critic --learn-logic-critic --reset-logic-actor --learn-logic-actor --total-timesteps 10000000 --actor-mode logic --reward-fn goal_only --env-frameskip 1 --env-max-ep-steps 3000"
ORACLE="fixed-static"
ORACLE_RESOLUTION=49
ENV_NAME="skiing"
STATIC_PREDICATES="true left_oriented right_oriented straight_oriented"
HAS_DEMO="0"
RULES="oriented_v1"

EXTRA_ARGS="--concept-coef 0.1 ${BASE_ARGS}"

NR="477_1st_gpt_cc0_1"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="477_1st_cl_cc0_1"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 0.3 ${BASE_ARGS}"

NR="477_1st_gpt_cc0_3"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="477_1st_cl_cc0_3"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 1.0 ${BASE_ARGS}"

NR="477_1st_gpt_cc1_0"
ORACLE_PROGRAM="chatgpt_4o_v7"
run_experiment

NR="477_1st_cl_cc1_0"
ORACLE_PROGRAM="claude_4sonnet_v7"
run_experiment

RULES="small"
HAS_DEMO="1"

# ======================================================================================================================
# ABLATIONS
# 1st stage; Different Concept Coefficients; No Concept Annealing

# ----------------------------------------------------------------------------------------------------------------------
# Kangaroo
# ----------------------------------------------------------------------------------------------------------------------
BASE_ARGS="--reset-neural-component --reset-blending-weights --reset-logic-critic --learn-logic-critic --reset-logic-actor --learn-logic-actor --total-timesteps 10000000 --actor-mode logic --extra-env-modifications disable_monkeys --reward-fn goal_only --env-frameskip 4 --env-max-ep-steps 3000"
ORACLE="fixed-static"
ORACLE_RESOLUTION=49
ENV_NAME="kangaroo"
STATIC_PREDICATES="close_by_monkey close_by_throwncoconut nothing_around"

EXTRA_ARGS="--concept-coef 0.01 ${BASE_ARGS}"

NR="209_gpt_cc0_01"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="209_cl_cc0_01"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 0.03 ${BASE_ARGS}"

NR="209_gpt_cc0_03"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="209_cl_cc0_03"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 0.1 ${BASE_ARGS}"

NR="209_gpt_cc0_1"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="209_cl_cc0_1"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 0.3 ${BASE_ARGS}"

NR="209_gpt_cc0_3"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="209_cl_cc0_3"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 1.0 ${BASE_ARGS}"

NR="209_gpt_cc1_0"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="209_cl_cc1_0"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

# ----------------------------------------------------------------------------------------------------------------------
# Seaquest
# ----------------------------------------------------------------------------------------------------------------------
BASE_ARGS="--reset-neural-component --reset-blending-weights --reset-logic-critic --learn-logic-critic --reset-logic-actor --learn-logic-actor --total-timesteps 10000000 --actor-mode logic --extra-env-modifications disable_enemies --reward-fn goal_only --env-frameskip 4 --env-max-ep-steps 3000"
ORACLE="fixed-static"
ORACLE_RESOLUTION=49
ENV_NAME="seaquest"
STATIC_PREDICATES="visible_diver full_divers not_full_divers oxygen_low close_by_missile close_by_enemy"

EXTRA_ARGS="--concept-coef 0.01 ${BASE_ARGS}"

NR="309_gpt_cc0_01"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="309_cl_cc0_01"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 0.03 ${BASE_ARGS}"

NR="309_gpt_cc0_03"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="309_cl_cc0_03"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 0.1 ${BASE_ARGS}"

NR="309_gpt_cc0_1"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="309_cl_cc0_1"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 0.3 ${BASE_ARGS}"

NR="309_gpt_cc0_3"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="309_cl_cc0_3"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment

EXTRA_ARGS="--concept-coef 1.0 ${BASE_ARGS}"

NR="309_gpt_cc1_0"
ORACLE_PROGRAM="chatgpt_4o_v7"
#run_experiment

NR="309_cl_cc1_0"
ORACLE_PROGRAM="claude_4sonnet_v7"
#run_experiment
# ======================================================================================================================


# ======================================================================================================================
# 2nd stage
# ======================================================================================================================

# Hybrid
# Neural Critic / Actor; Blender  = init. randomly; finetune
# Logic Critic / Actor            = init. from 1st experiment; frozen
# Valuation Functions             = init. from 1st experiment; frozen
# Oracle                          = enabled (blender predicates)

# ======================================================================================================================
# 2nd stage; c_CA=0.3; CA-annealing; Different Reward Functions

# ----------------------------------------------------------------------------------------------------------------------
# Kangaroo
# ----------------------------------------------------------------------------------------------------------------------

ENV_NAME="kangaroo"
STATIC_PREDICATES="nothing_around"
END_SEED=516
START_SEED=515
ORACLE="fixed-static"
ORACLE_RESOLUTION=49

# ChatGPT
BASE_ARGS="--anneal-concept-coef --concept-coef 0.3 --reset-neural-component --learn-neural-component --reset-blending-weights --learn-blending-weights --logic-critic-path out_val/runs/mlp_257_gpt_cc0_3_s1 --logic-actor-path out_val/runs/mlp_257_gpt_cc0_3_s1 --valuation-model-path out_val/runs/mlp_257_gpt_cc0_3_s1 --freeze-valuation-model --total-timesteps 60000000 --actor-mode hybrid --env-frameskip 1"
ORACLE_PROGRAM="chatgpt_4o_v7"

EXTRA_ARGS="--reward-fn default ${BASE_ARGS}"
NR="216_gpt_def"
#run_experiment

EXTRA_ARGS="--reward-fn goal_only ${BASE_ARGS}"
NR="216_gpt_go"
#run_experiment

# Claude
BASE_ARGS="--anneal-concept-coef --concept-coef 0.3 --reset-neural-component --learn-neural-component --reset-blending-weights --learn-blending-weights --logic-critic-path out_val/runs/mlp_257_cl_cc0_3_s1 --logic-actor-path out_val/runs/mlp_257_cl_cc0_3_s1 --valuation-model-path out_val/runs/mlp_257_cl_cc0_3_s1 --freeze-valuation-model --total-timesteps 60000000 --actor-mode hybrid --env-frameskip 1"
ORACLE_PROGRAM="claude_4sonnet_v7"

EXTRA_ARGS="--reward-fn default ${BASE_ARGS}"
NR="216_cl_def"
#run_experiment

EXTRA_ARGS="--reward-fn goal_only ${BASE_ARGS}"
NR="216_cl_go"
#run_experiment

# ----------------------------------------------------------------------------------------------------------------------
# Seaquest
# ----------------------------------------------------------------------------------------------------------------------
ENV_NAME="seaquest"
STATIC_PREDICATES="visible_diver full_divers not_full_divers oxygen_low"
END_SEED=1027
START_SEED=1026
ORACLE="fixed-static"
ORACLE_RESOLUTION=49

# ChatGPT
BASE_ARGS="--anneal-concept-coef --concept-coef 0.3 --reset-neural-component --learn-neural-component --reset-blending-weights --learn-blending-weights --logic-critic-path out_val/runs/mlp_309_gpt_s0 --logic-actor-path out_val/runs/mlp_309_gpt_s0 --valuation-model-path out_val/runs/mlp_309_gpt_s0 --freeze-valuation-model --total-timesteps 60000000 --actor-mode hybrid --env-frameskip 1"
ORACLE_PROGRAM="chatgpt_4o_v7"

EXTRA_ARGS="--reward-fn default ${BASE_ARGS}"
NR="316_gpt_def"
#run_experiment

EXTRA_ARGS="--reward-fn goal_only ${BASE_ARGS}"
NR="316_gpt_go"
#run_experiment

# Claude
BASE_ARGS="--anneal-concept-coef --concept-coef 0.3 --reset-neural-component --learn-neural-component --reset-blending-weights --learn-blending-weights --logic-critic-path out_val/runs/mlp_309_cl_s0 --logic-actor-path out_val/runs/mlp_309_cl_s0 --valuation-model-path out_val/runs/mlp_309_cl_s0 --freeze-valuation-model --total-timesteps 60000000 --actor-mode hybrid --env-frameskip 1"
ORACLE_PROGRAM="claude_4sonnet_v7"

EXTRA_ARGS="--reward-fn default ${BASE_ARGS}"
NR="316_cl_def"
#run_experiment

EXTRA_ARGS="--reward-fn goal_only ${BASE_ARGS}"
NR="316_cl_go"
#run_experiment
# ======================================================================================================================
