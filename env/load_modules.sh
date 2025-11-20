#!/bin/bash
# load_modules.sh
# Helper to load modules for building/running the miniWeather mini-app on
# Magic Castle/Alliance clusters. Adjust module names to match env/modules.txt.

module purge
module load gcc/XX.Y
module load openmpi/XX.Y
module load python/3.X
module load matplotlib/3.X || true  # ignore if matplotlib comes with python
module load likwid/XX.Y || true      # optional profiling support

# Default to all available cores unless overridden; users can change when using Slurm.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}

echo "Loaded modules and set OMP_NUM_THREADS=${OMP_NUM_THREADS}"
