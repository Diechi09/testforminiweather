#!/bin/bash
# miniweather_weak.sbatch
# Weak scaling Slurm script; per-rank nx is fixed, global nx grows with node count.

#SBATCH -J miniweather_weak
#SBATCH -N 4                  # TODO: sweep node counts
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH -t 00:30:00
#SBATCH -A <account>
#SBATCH -o weak_%j.out

cd $SLURM_SUBMIT_DIR
source ../env/load_modules.sh

# Define per-rank size; total nx = nx_per_rank * ntasks
NX_PER_RANK=256
NZ=512
STEPS=100
TOTAL_NTASKS=$((SLURM_NNODES * ${SLURM_NTASKS_PER_NODE:-16}))
TOTAL_NX=$((NX_PER_RANK * TOTAL_NTASKS))

echo "Computed global nx=${TOTAL_NX} for ${TOTAL_NTASKS} ranks"

srun ./miniweather_mpi_omp --nx ${TOTAL_NX} --nz ${NZ} --steps ${STEPS} --output ../results/weak_${SLURM_JOB_ID}.csv

# Collect outputs into results/weak_scaling.csv for plotting.
