#!/bin/bash
# profile_cpu.sbatch
# Example profiling script using perf (or likwid if available) on a single run.

#SBATCH -J miniweather_profile
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH -t 00:15:00
#SBATCH -A <account>
#SBATCH -o profile_%j.out

cd $SLURM_SUBMIT_DIR
source ../env/load_modules.sh

# Use perf stat as a portable default; replace with likwid-perfctr if available.
srun perf stat -d ./miniweather_mpi_omp --nx 512 --nz 256 --steps 50 --output ../results/profile_${SLURM_JOB_ID}.csv

# Store perf output (stdout) in results/profile_cpu_<jobid>.txt if desired.
