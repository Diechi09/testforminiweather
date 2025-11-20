#!/bin/bash
#SBATCH -J miniweather-prof-gpu
#SBATCH -A <account>
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -t 00:10:00
#SBATCH -o profile-%j.out

# Profiling template: adapt to available tools (perf/likwid/nsys/nvprof).
source env/load_modules.sh
cd $SLURM_SUBMIT_DIR/src
make gpu

NX=1024
NZ=512
STEPS=100
OUTPUT=$SLURM_SUBMIT_DIR/results/profile_${SLURM_JOB_ID}.csv

# Example using NVIDIA Nsight Systems if available; replace with perf/likwid as needed.
if command -v nsys &> /dev/null; then
  srun --mpi=pmix_v3 nsys profile -o $SLURM_SUBMIT_DIR/results/nsys_${SLURM_JOB_ID} \
    ./miniweather_mpi_cuda --nx ${NX} --nz ${NZ} --steps ${STEPS} --output ${OUTPUT} --gpu
else
  srun --mpi=pmix_v3 ./miniweather_mpi_cuda --nx ${NX} --nz ${NZ} --steps ${STEPS} --output ${OUTPUT} --gpu
fi
