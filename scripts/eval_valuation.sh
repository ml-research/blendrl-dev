#!/bin/bash

eval_valuation()
{
DOCKER_CONTAINER_NAME="eval_valuation_${ENV_NAME}"
AGENT_PATH="models/${ENV_NAME}_demo"

# Navigate to the parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Stop and remove container if still running
docker stop "${DOCKER_CONTAINER_NAME}" 2>/dev/null || true
docker rm "${DOCKER_CONTAINER_NAME}" 2>/dev/null || true

# Include baseline
EXTRA_ARGS=""
if [ "${INCLUDE_BASELINE}" == "1" ]; then
  EXTRA_ARGS="--include-baseline --baseline-actor-modes ${BASELINE_ACTOR_MODES}"
fi

# Run docker container
docker run \
  -v $PARENT_DIR:/app \
  --ipc=host --ulimit stack=67108864 \
  --rm \
  --name "${DOCKER_CONTAINER_NAME}" \
  blendrl:base \
  bash -c "cd /app & python eval_valuation.py --exp-names ${EXP_NAMES[*]} --agent-path ${AGENT_PATH} --exp-actor-modes ${ACTOR_MODES} ${EXTRA_ARGS}"
}

# ======================================================================================================================
# KANGAROO
# ======================================================================================================================

ENV_NAME="kangaroo"

# ----------------------------------------------------------------------------------------------------------------------
ACTOR_MODES="logic"
BASELINE_ACTOR_MODES="logic"
INCLUDE_BASELINE="1"

# Kangaroo (Baselines Logic + BlendRL)
EXP_NAMES=("mlp_213_s513" "mlp_213_s514" "mlp_213_s515")
#eval_valuation
INCLUDE_BASELINE="0"

# Kangaroo (1st stage)
EXP_NAMES=("mlp_257_cl_cc0_3_s0" "mlp_257_cl_cc0_3_s1" "mlp_257_cl_cc0_3_s2" "mlp_257_gpt_cc0_3_s0" "mlp_257_gpt_cc0_3_s1" "mlp_257_gpt_cc0_3_s2")
#eval_valuation

# Kangaroo (1st stage; Different Concept Coefficients; With Concept Annealing)
EXP_NAMES=("mlp_209_cl_s0" "mlp_209_cl_s1" "mlp_209_cl_s2" "mlp_209_gpt_s0" "mlp_209_gpt_s1" "mlp_209_gpt_s2" "mlp_257_cl_cc0_1_s0" "mlp_257_cl_cc0_1_s1" "mlp_257_cl_cc0_1_s2" "mlp_257_gpt_cc0_1_s0" "mlp_257_gpt_cc0_1_s1" "mlp_257_gpt_cc0_1_s2" "mlp_257_cl_cc0_3_s0" "mlp_257_cl_cc0_3_s1" "mlp_257_cl_cc0_3_s2" "mlp_257_gpt_cc0_3_s0" "mlp_257_gpt_cc0_3_s1" "mlp_257_gpt_cc0_3_s2" "mlp_257_cl_cc1_0_s0" "mlp_257_cl_cc1_0_s1" "mlp_257_cl_cc1_0_s2" "mlp_257_gpt_cc1_0_s0" "mlp_257_gpt_cc1_0_s1" "mlp_257_gpt_cc1_0_s2")
#eval_valuation

# Kangaroo (1st stage; Different Concept Coefficients; No Concept Annealing)
EXP_NAMES=("mlp_209_cl_cc0_03_s0" "mlp_209_cl_cc0_03_s1" "mlp_209_cl_cc0_03_s2" "mlp_209_gpt_cc0_03_s0" "mlp_209_gpt_cc0_03_s1" "mlp_209_gpt_cc0_03_s2" "mlp_209_cl_cc0_1_s0" "mlp_209_cl_cc0_1_s1" "mlp_209_cl_cc0_1_s2" "mlp_209_gpt_cc0_1_s0" "mlp_209_gpt_cc0_1_s1" "mlp_209_gpt_cc0_1_s2" "mlp_209_cl_cc0_3_s0" "mlp_209_cl_cc0_3_s1" "mlp_209_cl_cc0_3_s2" "mlp_209_gpt_cc0_3_s0" "mlp_209_gpt_cc0_3_s1" "mlp_209_gpt_cc0_3_s2" "mlp_209_cl_cc1_0_s0" "mlp_209_cl_cc1_0_s1" "mlp_209_cl_cc1_0_s2" "mlp_209_gpt_cc1_0_s0" "mlp_209_gpt_cc1_0_s1" "mlp_209_gpt_cc1_0_s2")
#eval_valuation

# ----------------------------------------------------------------------------------------------------------------------
ACTOR_MODES="neural"

# Kangaroo (Baseline Neural)
EXP_NAMES=("mlp_214_s513" "mlp_214_s514" "mlp_214_s515")
#eval_valuation

# ----------------------------------------------------------------------------------------------------------------------
ACTOR_MODES="logic_oracle"

# Kangaroo (Baselines ChatGPT + Claude)
EXP_NAMES=("mlp_257_cl_cc0_3_s1" "mlp_257_gpt_cc0_3_s1")
#eval_valuation


# ----------------------------------------------------------------------------------------------------------------------
# 2nd stage (Neural Baseline)
ACTOR_MODES="neural"

EXP_NAMES=("mlp_212_s513" "mlp_212_s514" "mlp_212_s515")
#eval_valuation


# ----------------------------------------------------------------------------------------------------------------------
# 2nd stage
ACTOR_MODES="hybrid"
BASELINE_ACTOR_MODES="hybrid"
INCLUDE_BASELINE="1"

EXP_NAMES=("mlp_216_gpt_def_s513" "mlp_216_gpt_def_s514" "mlp_216_gpt_def_s515" "mlp_216_cl_def_s513" "mlp_216_cl_def_s514" "mlp_216_cl_def_s515")
#eval_valuation
INCLUDE_BASELINE="0"

# ----------------------------------------------------------------------------------------------------------------------
# 2nd stage (goal only
EXP_NAMES=("mlp_216_gpt_go_s513" "mlp_216_gpt_go_s514" "mlp_216_cl_go_s513" "mlp_216_cl_go_s514")
#eval_valuation

# ----------------------------------------------------------------------------------------------------------------------
# 2nd stage (No CA-annealing)
EXP_NAMES=("mlp_252_cl_cc0_1_s515" "mlp_252_cl_cc0_3_s515" "mlp_252_cl_cc1_0_s515" "mlp_252_cl_cc3_0_s515")
#eval_valuation

# ======================================================================================================================
# SEAQUEST
# ======================================================================================================================

ENV_NAME="seaquest"

# ----------------------------------------------------------------------------------------------------------------------
ACTOR_MODES="logic"
BASELINE_ACTOR_MODES="logic"
INCLUDE_BASELINE="1"

# Seaquest (Baselines Logic + BlendRL)
EXP_NAMES=("mlp_313_s1024" "mlp_313_s1025" "mlp_313_s1026")
#eval_valuation
INCLUDE_BASELINE="0"

# Seaquest (1st stage)
EXP_NAMES=("mlp_309_cl_s0" "mlp_309_cl_s1" "mlp_309_cl_s2" "mlp_309_gpt_s0" "mlp_309_gpt_s1" "mlp_309_gpt_s2")
#eval_valuation

# Seaquest (1st stage; Different Concept Coefficients; With Concept Annealing)
EXP_NAMES=("mlp_309_cl_s0" "mlp_309_cl_s1" "mlp_309_cl_s2" "mlp_309_gpt_s0" "mlp_309_gpt_s1" "mlp_309_gpt_s2" "mlp_357_cl_cc0_1_s0" "mlp_357_cl_cc0_1_s1" "mlp_357_cl_cc0_1_s2" "mlp_357_gpt_cc0_1_s0" "mlp_357_gpt_cc0_1_s1" "mlp_357_gpt_cc0_1_s2" "mlp_357_cl_cc0_3_s0" "mlp_357_cl_cc0_3_s1" "mlp_357_cl_cc0_3_s2" "mlp_357_gpt_cc0_3_s0" "mlp_357_gpt_cc0_3_s1" "mlp_357_gpt_cc0_3_s2" "mlp_357_cl_cc1_0_s0" "mlp_357_cl_cc1_0_s1" "mlp_357_cl_cc1_0_s2" "mlp_357_gpt_cc1_0_s0" "mlp_357_gpt_cc1_0_s1" "mlp_357_gpt_cc1_0_s2")
#eval_valuation

# Seaquest (1st stage; Different Concept Coefficients; No Concept Annealing)
EXP_NAMES=("mlp_309_cl_cc0_03_s0" "mlp_309_cl_cc0_03_s1" "mlp_309_cl_cc0_03_s2" "mlp_309_gpt_cc0_03_s0" "mlp_309_gpt_cc0_03_s1" "mlp_309_gpt_cc0_03_s2" "mlp_309_cl_cc0_1_s0" "mlp_309_cl_cc0_1_s1" "mlp_309_cl_cc0_1_s2" "mlp_309_gpt_cc0_1_s0" "mlp_309_gpt_cc0_1_s1" "mlp_309_gpt_cc0_1_s2" "mlp_309_cl_cc0_3_s0" "mlp_309_cl_cc0_3_s1" "mlp_309_cl_cc0_3_s2" "mlp_309_gpt_cc0_3_s0" "mlp_309_gpt_cc0_3_s1" "mlp_309_gpt_cc0_3_s2" "mlp_309_cl_cc1_0_s0" "mlp_309_cl_cc1_0_s1" "mlp_309_cl_cc1_0_s2" "mlp_309_gpt_cc1_0_s0" "mlp_309_gpt_cc1_0_s1" "mlp_309_gpt_cc1_0_s2")
#eval_valuation

# ----------------------------------------------------------------------------------------------------------------------
ACTOR_MODES="neural"

# Seaquest (Baseline Neural)
EXP_NAMES=("mlp_314_s1024" "mlp_314_s1025" "mlp_314_s1026")
#eval_valuation

# ----------------------------------------------------------------------------------------------------------------------
ACTOR_MODES="logic_oracle"

# Seaquest (Baselines ChatGPT + Claude)
EXP_NAMES=("mlp_309_cl_s0" "mlp_309_gpt_s0")
#eval_valuation

# ----------------------------------------------------------------------------------------------------------------------
# 2nd stage (Neural Baseline)
ACTOR_MODES="neural"

EXP_NAMES=("mlp_312_s1024" "mlp_312_s1025" "mlp_312_s1026")
#eval_valuation

# ----------------------------------------------------------------------------------------------------------------------
# 2nd stage
ACTOR_MODES="hybrid"
BASELINE_ACTOR_MODES="hybrid"
INCLUDE_BASELINE="1"

EXP_NAMES=("mlp_316_gpt_def_s1024" "mlp_316_gpt_def_s1025" "mlp_316_gpt_def_s1026" "mlp_316_cl_def_s1024" "mlp_316_cl_def_s1025" "mlp_316_cl_def_s1026")
#eval_valuation
INCLUDE_BASELINE="0"

# ----------------------------------------------------------------------------------------------------------------------
# 2nd stage (goal only
EXP_NAMES=("mlp_316_gpt_go_s1024" "mlp_316_gpt_go_s1025" "mlp_316_cl_go_s1024" "mlp_316_cl_go_s1025")
#eval_valuation

# ----------------------------------------------------------------------------------------------------------------------
# 2nd stage (No CA-annaeling)
EXP_NAMES=("mlp_252_cl_cc0_1_s515" "mlp_252_cl_cc0_3_s515" "mlp_252_cl_cc1_0_s515" "mlp_252_cl_cc3_0_s515")
#eval_valuation