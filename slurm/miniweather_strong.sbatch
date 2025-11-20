#!/bin/bash
# miniweather_strong.sbatch
# Strong scaling Slurm script for the miniWeather mini-app on CPU nodes.
# Run multiple jobs with different -N values and combine the CSV rows into
# results/strong_scaling.csv after collecting outputs.

#SBATCH -J miniweather_strong
#SBATCH -N 4                  # TODO: sweep 1,2,4,... nodes
#SBATCH --ntasks-per-node=16  # adjust to match physical cores
#SBATCH --cpus-per-task=2     # OpenMP threads per rank
#SBATCH -t 00:30:00
#SBATCH -A <account>
#SBATCH -o strong_%j.out

cd $SLURM_SUBMIT_DIR
source ../env/load_modules.sh

srun ./miniweather_mpi_omp --nx 1024 --nz 512 --steps 100 --output ../results/strong_${SLURM_JOB_ID}.csv

# After running multiple jobs, aggregate into results/strong_scaling.csv and
# plot with: python ../src/plot_scaling.py
