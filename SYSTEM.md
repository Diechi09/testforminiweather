# SYSTEM.md

Document the hardware/software configuration for reproducibility.

## Hardware
- Node type: TODO (e.g., GPU nodes on Magic Castle)
- CPU: TODO (model, cores per socket, sockets per node)
- GPU: TODO (e.g., NVIDIA A100, count per node 4/8)
- Memory: TODO (per node GB)
- Interconnect: TODO (InfiniBand HDR/NDR or equivalent)

## Software
- Compiler: TODO (gcc/clang + version)
- MPI: TODO (OpenMPI/MPICH + version)
- CUDA: TODO (driver + toolkit version)
- Python/matplotlib: TODO
- Profilers: TODO (nsys/nvprof/perf/likwid)

## Runtime environment
- OMP_NUM_THREADS: TODO
- CUDA_VISIBLE_DEVICES: TODO (if manually set)
- Slurm: mention job submission flags used for scaling and profiling.

Keep this file updated for every experimental campaign.
