# MiniWeather MPI/CUDA Mini-App

This repository provides a hybrid MPI miniWeather-style mini-application that can
run either on CPU (MPI + OpenMP) or on NVIDIA GPUs (MPI + CUDA). It implements a
2D advection-diffusion stencil with halo exchanges and timing hooks for scaling
studies. The layout targets coursework on Magic Castle/Alliance clusters using
Slurm and EESSI modules, with ready-to-use Slurm scripts for 4–8 GPU nodes.
If CUDA devices are absent at runtime the code automatically falls back to the
CPU path while retaining MPI+OpenMP behavior.

The executable writes CSV rows that include communication and compute averages
per step plus a communication fraction, making bottleneck attribution and
efficiency plots straightforward (see `results/` for sample data and
`src/plot_scaling.py` for automated plots).

Key directories:
- `src/`: C++ source (`miniweather_mpi_omp.cpp`), Makefile, and plotting helper.
- `env/`: module list, loader script, and Apptainer definition for portability.
- `slurm/`: example batch scripts for scaling/profiling plus a container wrapper.
- `data/`: placeholder for datasets (synthetic by default).
- `results/`: location to store CSVs, plots, and profiling logs.
- `docs/`: outlines for the short paper, EuroHPC proposal, and pitch deck.

See `reproduce.md` for step-by-step build and run instructions (CPU and GPU).

## Local frontend viewer

A lightweight Flask frontend is included for quick demos on your laptop. After
running a few experiments (or relying on the built-in sample rows), start:

```
python -m venv .venv && source .venv/bin/activate
pip install flask matplotlib pandas
python src/frontend.py
```

Then open http://127.0.0.1:5000/ to browse strong/weak scaling CSVs and plots
from the `results/` folder without needing Slurm. The frontend uses the same
CSV schema as the mini-app and defaults to bundled sample rows so the dashboard
renders even before you run on the cluster.
