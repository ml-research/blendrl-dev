#!/bin/bash

plot_experiment()
{
  # Docker GPU flag
  if [ "$CUDA_DEVICES" == "all" ]; then
    DOCKER_GPUS="all"
  else
    DOCKER_GPUS="\"device=$CUDA_DEVICES\""
  fi

  # Animations
  if [ "${SAVE_ANIMATIONS}" == "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --save-animations"
  fi

  # Navigate to the parent directory
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PARENT_DIR="$(dirname "$SCRIPT_DIR")"

  for EXP_NAME in "${EXP_NAMES[@]}"; do

    DOCKER_CONTAINER_NAME="plot_predicate_valuation_${EXP_NAME}"

    echo "Plotting ${EXP_NAME}"

    # Run docker container in background
    nohup docker run \
      -v $PARENT_DIR:/app \
      --gpus ${DOCKER_GPUS} \
      --ipc=host --ulimit stack=67108864 \
      --rm \
      --name "${DOCKER_CONTAINER_NAME}" \
      blendrl:base \
      bash -c "cd /app & python -m plot.plot_predicate_valuation --exp-name ${EXP_NAME} --predicate-name ${PREDICATES} --position ${POSITIONS} --logic-actor-positions ${LOGIC_ACTOR_POSITIONS} --skip ${SKIP} --plots ${PLOTS} ${EXTRA_ARGS}" > ${SCRIPT_DIR}/plot_predicate_valuation.log 2>&1 &

  done
}

CUDA_DEVICES="2"
SKIP=5
SAVE_ANIMATIONS="0"

# ======================================================================================================================
# Kangaroo
# ======================================================================================================================
EXTRA_ARGS="--contour-objects Ladder Platform"

# Kangaroo Baselines
EXP_NAMES=("mlp_207_s0" "mlp_207_s1" "mlp_207_s2" "mlp_208_s0" "mlp_208_s1" "mlp_208_s2")
PLOTS="logic_critic valuation valuation_overlay"
PREDICATES="left_of_ladder right_of_ladder on_ladder"
POSITIONS="Ladder_1 Ladder_2 Ladder_3"
#plot_experiment
PLOTS="valuation"
PREDICATES="close_by_monkey close_by_throwncoconut"
POSITIONS="5"
#plot_experiment

# Kangaroo Baseline E Logic
EXP_NAMES=("mlp_213_s513" "mlp_213_s514" "mlp_213_s515")
PLOTS="logic_critic valuation valuation_overlay actions actions_overlay"
PREDICATES="left_of_ladder right_of_ladder on_ladder"
POSITIONS="Ladder_1 Ladder_2 Ladder_3"
#plot_experiment

# Kangaroo 1st stage
EXP_NAMES=("mlp_209_cl_s0" "mlp_209_cl_s1" "mlp_209_cl_s2" "mlp_209_gpt_s0" "mlp_209_gpt_s1" "mlp_209_gpt_s2")
PLOTS="logic_critic valuation valuation_overlay oracle_diff oracle_diff_overlay actions actions_overlay"
PREDICATES="left_of_ladder right_of_ladder on_ladder"
POSITIONS="Ladder_1 Ladder_2 Ladder_3"
#plot_experiment

# Kangaroo 2nd stage
EXP_NAMES=("mlp_210_cl_s513" "mlp_210_cl_s514" "mlp_210_cl_s515" "mlp_210_gpt_s513" "mlp_210_gpt_s514" "mlp_210_gpt_s515")
PLOTS="valuation oracle_diff"
PREDICATES="close_by_monkey close_by_throwncoconut"
POSITIONS="5"
#plot_experiment

# Kangaroo Ablations (Reward functions)
EXP_NAMES=("mlp_251_go_s515" "mlp_251_wl_s515" "mlp_251_wdp_s515" "mlp_251_wladp_s515" "mlp_251_defwdp_s515")
PLOTS="valuation"
PREDICATES="close_by_monkey close_by_throwncoconut"
POSITIONS="5"
#plot_experiment

# Kangaroo Ablations (1st stage; Different Concept Coefficients; No Concept Annealing)
EXP_NAMES=("mlp_209_cl_cc0_03_s0" "mlp_209_cl_cc0_03_s1" "mlp_209_cl_cc0_03_s2" "mlp_209_gpt_cc0_03_s0" "mlp_209_gpt_cc0_03_s1" "mlp_209_gpt_cc0_03_s2" "mlp_209_cl_cc0_1_s0" "mlp_209_cl_cc0_1_s1" "mlp_209_cl_cc0_1_s2" "mlp_209_gpt_cc0_1_s0" "mlp_209_gpt_cc0_1_s1" "mlp_209_gpt_cc0_1_s2" "mlp_209_cl_cc0_3_s0" "mlp_209_cl_cc0_3_s1" "mlp_209_cl_cc0_3_s2" "mlp_209_gpt_cc0_3_s0" "mlp_209_gpt_cc0_3_s1" "mlp_209_gpt_cc0_3_s2" "mlp_209_cl_cc1_0_s0" "mlp_209_cl_cc1_0_s1" "mlp_209_cl_cc1_0_s2" "mlp_209_gpt_cc1_0_s0" "mlp_209_gpt_cc1_0_s1" "mlp_209_gpt_cc1_0_s2")
PLOTS="logic_critic valuation valuation_overlay oracle_diff oracle_diff_overlay actions actions_overlay"
PREDICATES="left_of_ladder right_of_ladder on_ladder"
POSITIONS="Ladder_1 Ladder_2 Ladder_3"
#plot_experiment

# Kangaroo Ablations (1st stage; Different Concept Coefficients; With Concept Annealing)
EXP_NAMES=("mlp_257_cl_cc0_1_s0" "mlp_257_cl_cc0_1_s1" "mlp_257_cl_cc0_1_s2" "mlp_257_gpt_cc0_1_s0" "mlp_257_gpt_cc0_1_s1" "mlp_257_gpt_cc0_1_s2" "mlp_257_cl_cc0_3_s0" "mlp_257_cl_cc0_3_s1" "mlp_257_cl_cc0_3_s2" "mlp_257_gpt_cc0_3_s0" "mlp_257_gpt_cc0_3_s1" "mlp_257_gpt_cc0_3_s2" "mlp_257_cl_cc1_0_s0" "mlp_257_cl_cc1_0_s1" "mlp_257_cl_cc1_0_s2" "mlp_257_gpt_cc1_0_s0" "mlp_257_gpt_cc1_0_s1" "mlp_257_gpt_cc1_0_s2")
PLOTS="logic_critic valuation valuation_overlay oracle_diff oracle_diff_overlay actions actions_overlay"
PREDICATES="left_of_ladder right_of_ladder on_ladder"
POSITIONS="Ladder_1 Ladder_2 Ladder_3"
#plot_experiment

# Kangaroo Ablations (2nd stage + Concept coefficients)
EXP_NAMES=("mlp_252_cl_cc0_1_s515" "mlp_252_cl_cc0_3_s515" "mlp_252_cl_cc1_0_s515" "mlp_252_cl_cc3_0_s515")
PLOTS="valuation oracle_diff"
PREDICATES="close_by_monkey close_by_throwncoconut"
POSITIONS="5"
#plot_experiment

# Kangaroo CA loss specific checkpoints
EXP_NAMES=("mlp_257_cl_cc0_3_s1")
PLOTS="valuation_overlay oracle_diff_overlay"
POSITIONS="Ladder_1 Ladder_2 Ladder_3"
PREDICATES="left_of_ladder"
EXTRA_ARGS="--contour-objects Ladder Platform --checkpoint 1330000 2350000"
#plot_experiment

# Kangaroo 2nd stage
EXP_NAMES=("mlp_216_gpt_def_s513" "mlp_216_cl_def_s513" "mlp_216_gpt_go_s513" "mlp_216_cl_go_s513")
PLOTS="valuation oracle_diff"
PREDICATES="close_by_monkey close_by_throwncoconut"
POSITIONS="5"
#plot_experiment

# Kangaroo 2nd stage (goal only)
EXP_NAMES=("mlp_216_gpt_go_s513" "mlp_216_gpt_go_s514" "mlp_216_cl_go_s513" "mlp_216_cl_go_s514")
PLOTS="valuation oracle_diff"
PREDICATES="close_by_monkey close_by_throwncoconut"
POSITIONS="5"
#plot_experiment

# ======================================================================================================================
# Seaquest
# ======================================================================================================================
EXTRA_ARGS=""
POSITIONS="5"

# Seaquest Baselines
EXP_NAMES=("mlp_307_s0" "mlp_307_s1" "mlp_307_s2" "mlp_308_s0" "mlp_308_s1" "mlp_308_s2")
PLOTS="valuation"
PREDICATES="left_of_diver right_of_diver higher_than_diver deeper_than_diver close_by_missile close_by_enemy"
#plot_experiment

# Seaquest Baseline E Logic
EXP_NAMES=("mlp_313_s1024" "mlp_313_s1025" "mlp_313_s1026")
PLOTS="valuation"
PREDICATES="left_of_diver right_of_diver higher_than_diver deeper_than_diver"
#plot_experiment

# Seaquest 1st stage
EXP_NAMES=("mlp_309_cl_s0" "mlp_309_cl_s1" "mlp_309_cl_s2" "mlp_309_gpt_s0" "mlp_309_gpt_s1" "mlp_309_gpt_s2")
PLOTS="valuation oracle_diff"
PREDICATES="left_of_diver right_of_diver higher_than_diver deeper_than_diver"
#plot_experiment

# Seaquest Ablations (1st stage; Different Concept Coefficients; No Concept Annealing)
EXP_NAMES=("mlp_309_cl_cc0_1_s0" "mlp_309_cl_cc0_1_s1" "mlp_309_cl_cc0_1_s2" "mlp_309_gpt_cc0_1_s0" "mlp_309_gpt_cc0_1_s1" "mlp_309_gpt_cc0_1_s2" "mlp_309_cl_cc0_01_s0" "mlp_309_cl_cc0_01_s1" "mlp_309_cl_cc0_01_s2" "mlp_309_gpt_cc0_01_s0" "mlp_309_gpt_cc0_01_s1" "mlp_309_gpt_cc0_01_s2")
PLOTS="valuation oracle_diff"
PREDICATES="left_of_diver right_of_diver higher_than_diver deeper_than_diver"
#plot_experiment

# Seaquest Ablations (1st stage; Different Concept Coefficients; With Concept Annealing)
EXP_NAMES=("mlp_357_cl_cc0_1_s0" "mlp_357_cl_cc0_1_s1" "mlp_357_cl_cc0_1_s2" "mlp_357_gpt_cc0_1_s0" "mlp_357_gpt_cc0_1_s1" "mlp_357_gpt_cc0_1_s2" "mlp_357_cl_cc0_01_s0" "mlp_357_cl_cc0_01_s1" "mlp_357_cl_cc0_01_s2" "mlp_357_gpt_cc0_01_s0" "mlp_357_gpt_cc0_01_s1" "mlp_357_gpt_cc0_01_s2")
PLOTS="valuation oracle_diff"
PREDICATES="left_of_diver right_of_diver higher_than_diver deeper_than_diver"
#plot_experiment

# Seaquest 2nd stage
#EXP_NAMES=("mlp_310_cl_s1024" "mlp_310_cl_s1025" "mlp_310_cl_s1026" "mlp_310_gpt_s1024" "mlp_310_gpt_s1025" "mlp_310_gpt_s1026")
EXP_NAMES=("mlp_316_gpt_def_s1024" "mlp_316_cl_def_s1024" "mlp_316_gpt_go_s1024" "mlp_316_cl_go_s1024")
PLOTS="valuation oracle_diff"
PREDICATES="close_by_missile close_by_enemy"
POSITIONS="5"
#plot_experiment

# Seaquest 2nd stage (goal only)
EXP_NAMES=("mlp_316_gpt_go_s1024" "mlp_316_gpt_go_s1025" "mlp_316_cl_go_s1024" "mlp_316_cl_go_s1025")
PLOTS="valuation oracle_diff"
PREDICATES="close_by_missile close_by_enemy"
POSITIONS="5"
#plot_experiment
