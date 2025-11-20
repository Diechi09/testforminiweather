#!/usr/bin/env bash
# Module helper for Magic Castle/Alliance-style clusters.
# Replace module names with site-specific versions; keep a record for reproducibility.

module purge
module load gcc/XX.Y openmpi/XX.Y cuda/YY.Z python/3.X matplotlib/3.X
# Add optional modules if available; otherwise, install via pip inside a venv
# for local plotting/frontend visualisation.
# module load flask/2.X pandas/1.X

# Set a sensible default for OpenMP threads; override per job if needed.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

# Optional: export CUDA_VISIBLE_DEVICES if you want to pin ranks manually.
# export CUDA_VISIBLE_DEVICES=0,1,2,3
