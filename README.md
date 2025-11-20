# MiniWeather MPI+OpenMP Mini-App

This repository provides a compact hybrid MPI + OpenMP miniWeather-style
mini-application implementing a 2D advection-diffusion stencil. It is structured
for coursework on Magic Castle/Alliance clusters with Slurm and includes
scaffolding for strong/weak scaling studies, profiling, and documentation.

Key directories:
- `src/`: C++ source (`miniweather_mpi_omp.cpp`), Makefile, and plotting helper.
- `env/`: module list, loader script, and Apptainer definition for portability.
- `slurm/`: example batch scripts for scaling/profiling plus a container wrapper.
- `data/`: placeholder for datasets (synthetic by default).
- `results/`: location to store CSVs, plots, and profiling logs.
- `docs/`: outlines for the short paper, EuroHPC proposal, and pitch deck.

See `reproduce.md` for step-by-step build and run instructions.
