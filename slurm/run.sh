#!/usr/bin/env bash
# Helper wrapper to execute commands inside the Apptainer image if present.
# Also works for GPU builds when the host has NVIDIA drivers and CUDA runtime.

set -euo pipefail

IMAGE=env/project.sif
if [[ ! -f ${IMAGE} ]]; then
  echo "Apptainer image ${IMAGE} not found; build with: apptainer build ${IMAGE} env/project.def"
fi

# Example usage (GPU):
#   ./run.sh apptainer exec --nv --bind $PWD:$PWD --pwd $PWD env/project.sif srun --gpus-per-task=1 ./miniweather_mpi_cuda --gpu

exec "$@"
