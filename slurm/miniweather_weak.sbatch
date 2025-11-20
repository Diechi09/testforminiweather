#!/bin/bash
#SBATCH -J miniweather-weak-gpu
#SBATCH -A <account>
#SBATCH -N 4                 # adjust for more/fewer GPU nodes
#SBATCH --ntasks-per-node=4  # one MPI rank per GPU
#SBATCH --gpus-per-node=4
#SBATCH -t 00:15:00
#SBATCH -o weak-%j.out

# Weak-scaling: per-rank problem size fixed, grow global size with node count.
# Example keeps NX_PER_RANK constant; total NX = NX_PER_RANK * ntasks.

source env/load_modules.sh
cd $SLURM_SUBMIT_DIR/src
make gpu

NX_PER_RANK=256
NZ=1024
STEPS=200
NTASKS=$SLURM_NTASKS
NX_TOTAL=$((NX_PER_RANK * NTASKS))
OUTPUT=$SLURM_SUBMIT_DIR/results/weak_${SLURM_JOB_ID}.csv

echo "Running weak scaling with NX_TOTAL=${NX_TOTAL} (per-rank ${NX_PER_RANK}), NZ=${NZ}, tasks=${NTASKS}" \
     "GPUs=${SLURM_GPUS}" > $SLURM_SUBMIT_DIR/results/weak_${SLURM_JOB_ID}.log

srun --mpi=pmix_v3 ./miniweather_mpi_cuda --nx ${NX_TOTAL} --nz ${NZ} --steps ${STEPS} --output ${OUTPUT} --gpu

# After multiple runs at different -N, merge CSV rows into results/weak_scaling.csv.
