# src/

This folder contains the hybrid MPI + OpenMP miniWeather-style mini-application
and helper scripts.

## Contents
- `miniweather_mpi_omp.cpp`: main C++ code implementing a 2D
  advection-diffusion stencil with MPI domain decomposition in the x direction
  and OpenMP parallelism across the local subdomain.
- `Makefile`: builds the executable via `make` (override `CXX`/`CXXFLAGS` if
  needed to match the cluster toolchain).
- `plot_scaling.py`: convenience script to plot strong/weak scaling results.

## Building
```bash
cd src
module load <compiler> <mpi>   # or use env/load_modules.sh
make
```
This generates `miniweather_mpi_omp` in the same directory.

## Running
A simple single-node run (two MPI ranks, four threads each) using `srun`:
```bash
srun -n 2 --cpus-per-task=4 ./miniweather_mpi_omp --nx 256 --nz 128 --steps 50 --output ../results/example.csv
```
Command-line arguments:
- `--nx <int>`: global grid size in x (default: 256)
- `--nz <int>`: global grid size in z (default: 128)
- `--steps <int>`: number of time steps (default: 100)
- `--output <path>`: CSV file to append run metadata (default:
  `results/run_summary.csv` relative to repo root)

Halo exchange occurs along x between left/right ranks; the code prints a global
sum every 10 steps to verify correctness and timing information at the end.
