# miniweather_mpi_omp / miniweather_mpi_cuda

This directory contains the hybrid MPI miniWeather-style mini-app and helpers.

## Building

CPU (MPI + OpenMP):
```bash
module load gcc openmpi  # adjust for your site
cd src
make cpu
```

GPU (MPI + CUDA, one GPU per MPI rank):
```bash
module load gcc openmpi cuda  # adjust versions
cd src
make gpu
```

The GPU binary is named `miniweather_mpi_cuda`; the CPU binary is
`miniweather_mpi_omp`. Both use the same source file and CLI.

## Running a quick sanity test

Single node, 1 rank, CPU:
```bash
srun -n 1 ./miniweather_mpi_omp --nx 128 --nz 64 --steps 20 --output results/cpu_test.csv
```

Single node, 1 rank bound to 1 GPU:
```bash
srun -n 1 --gpus-per-task=1 ./miniweather_mpi_cuda --nx 128 --nz 64 --steps 20 --output results/gpu_test.csv --gpu
```

## Command-line arguments
- `--nx`, `--nz`: global grid dimensions.
- `--steps`: number of time steps.
- `--output`: CSV path for summary results.
- `--gpu` / `--cpu`: select device when the binary was built with CUDA support.

## Notes
- MPI performs a 1D domain decomposition in the x-dimension; halos are exchanged
  between left/right neighbors.
- OpenMP is applied across the local 2D interior for the CPU path.
- The GPU path uses CUDA for the stencil and stages halo columns through host
  buffers for MPI exchange (CUDA-aware MPI can replace this staging if available).
