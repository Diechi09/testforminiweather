# SYSTEM.md

Template for documenting the hardware/software environment used in experiments.
Fill in the details before submitting results.

## Cluster / Node Details
- Cluster name: Magic Castle (Alliance)
- Node type(s): TODO (e.g., Intel Xeon Gold 63xx)
- Cores per node: TODO
- Memory per node: TODO
- Interconnect: TODO (e.g., InfiniBand HDR)

## Software Stack
- Compiler: TODO (e.g., gcc/11.3)
- MPI: TODO (e.g., openmpi/4.1)
- Python: TODO (for plotting)
- Profiling tools: TODO (perf, likwid)

## Environment Variables
- `OMP_NUM_THREADS`: TODO (e.g., 8)
- `OMP_PLACES` / `OMP_PROC_BIND`: TODO
- MPI settings (e.g., `UCX_NET_DEVICES`, `MPICH_GPU_SUPPORT_ENABLED`): TODO

## Notes
- Mention any deviations from defaults (e.g., non-uniform core binding,
  hyperthreading on/off).
- Include Slurm version and relevant scheduler options if noteworthy.
