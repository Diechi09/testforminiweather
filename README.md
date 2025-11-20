# MiniWeather MPI/CUDA Mini-App

This repository provides a hybrid MPI miniWeather-style mini-application that can
run either on CPU (MPI + OpenMP) or on NVIDIA GPUs (MPI + CUDA). It implements a
2D advection-diffusion stencil with halo exchanges and timing hooks for scaling
studies. The layout targets coursework on Magic Castle/Alliance clusters using
Slurm and EESSI modules, with ready-to-use Slurm scripts for 4–8 GPU nodes.
If CUDA devices are absent at runtime the code automatically falls back to the
CPU path while retaining MPI+OpenMP behavior.

Key directories:
- `src/`: C++ source (`miniweather_mpi_omp.cpp`), Makefile, and plotting helper.
- `env/`: module list, loader script, and Apptainer definition for portability.
- `slurm/`: example batch scripts for scaling/profiling plus a container wrapper.
- `data/`: placeholder for datasets (synthetic by default).
- `results/`: location to store CSVs, plots, and profiling logs.
- `docs/`: outlines for the short paper, EuroHPC proposal, and pitch deck.

See `reproduce.md` for step-by-step build and run instructions (CPU and GPU).
