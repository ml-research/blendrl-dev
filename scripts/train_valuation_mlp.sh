#!/bin/bash

# Navigate to the parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Run docker container
docker run -d --rm \
  -v $PARENT_DIR:/app \
  --gpus all \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  -e WANDB_TEAM="$WANDB_TEAM" \
  blendrl:base \
  bash -c "cd /app & python train_valuation.py --wandb-entity $WANDB_TEAM --env-name kangaroo --num-steps 128 --num-envs 5 --track"