#!/bin/bash
#SBATCH -J miniweather-strong-gpu
#SBATCH -A <account>
#SBATCH -N 4                 # adjust node count for scaling study
#SBATCH --ntasks-per-node=4  # one MPI rank per GPU
#SBATCH --gpus-per-node=4    # request 4 GPUs/node (use 8 if available)
#SBATCH -t 00:15:00
#SBATCH -o strong-%j.out

# Strong-scaling: fixed global grid, vary nodes/GPUs.
# Run several jobs with different -N/--ntasks-per-node to build strong_scaling.csv.

source env/load_modules.sh
cd $SLURM_SUBMIT_DIR/src
make gpu

# Global problem size held constant for strong scaling.
NX=2048
NZ=1024
STEPS=200
OUTPUT=$SLURM_SUBMIT_DIR/results/strong_${SLURM_JOB_ID}.csv

srun --mpi=pmix_v3 ./miniweather_mpi_cuda --nx ${NX} --nz ${NZ} --steps ${STEPS} --output ${OUTPUT} --gpu

# After multiple runs, aggregate the CSV rows into results/strong_scaling.csv
# and plot with: python src/plot_scaling.py
