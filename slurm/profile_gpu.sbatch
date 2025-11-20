#!/bin/bash
#SBATCH -J miniweather-gpu-prof
#SBATCH -A <account>
#SBATCH -N 1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:20:00
#SBATCH -o results/profile_gpu_%j.out

# GPU profiling template using Nsight Systems/Compute; adjust tools per site.
source env/load_modules.sh
cd $SLURM_SUBMIT_DIR/src
make gpu

OUTPUT=$SLURM_SUBMIT_DIR/results/nsys_${SLURM_JOB_ID}.qdrep
# Use nsys if available, otherwise just run the binary (fallback still records CSV)
if command -v nsys >/dev/null 2>&1; then
  srun --mpi=pmix_v3 --gpus-per-task=1 nsys profile -o ${OUTPUT%.*} ./miniweather_mpi_cuda --nx 1024 --nz 512 --steps 100 --output $SLURM_SUBMIT_DIR/results/profile_gpu_${SLURM_JOB_ID}.csv --gpu
else
  srun --mpi=pmix_v3 --gpus-per-task=1 ./miniweather_mpi_cuda --nx 1024 --nz 512 --steps 100 --output $SLURM_SUBMIT_DIR/results/profile_gpu_${SLURM_JOB_ID}.csv --gpu
fi

# Copy profiler output into results/ for later analysis.
