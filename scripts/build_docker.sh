#!/bin/bash

# Define image name and tag
IMAGE_NAME="blendrl"

# Navigate to the parent directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

build_docker_target() {
  local image_tag="$1"

  # Build the Docker image
  echo "Building Docker image '$IMAGE_NAME:$image_tag'"
  echo "Docker Context: ${PARENT_DIR}"
  docker build --no-cache --target $image_tag -t "$IMAGE_NAME:$image_tag" "$PARENT_DIR"

  # Build complete
  if [ $? -eq 0 ]; then
    echo "Docker image '$IMAGE_NAME:$image_tag' built successfully!"
  else
    echo "Docker build failed."
    exit 1
  fi
}


# Build the base Docker image
build_docker_target "base"
