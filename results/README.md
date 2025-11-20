# results/

Store experiment outputs here.

- Strong scaling CSV: `results/strong_scaling.csv` (aggregate from multiple strong-scaling job outputs). Columns include timing breakdowns (`avg_comm_per_step`, `avg_compute_per_step`, `comm_fraction`) for bottleneck attribution plus `throughput_mcells_s` for efficiency.
- Weak scaling CSV: `results/weak_scaling.csv` (same schema as strong-scaling; keeps per-rank work roughly constant).
- Profiling logs: `results/profile_cpu_<details>.txt`, `results/nsys_<jobid>.qdrep`, etc. for GPU profiling. Add Nsight Compute summaries or perf/likwid counters here for paper/proposal evidence.

Run `python src/plot_scaling.py` after collecting data to generate
`strong_scaling.(png|svg)` and `weak_scaling.(png|svg)` in this folder.
`strong_scaling.*` includes a third panel highlighting communication vs compute per step.

For local demos without Slurm, launch the Flask frontend and open the default
address: `python src/frontend.py` then browse http://127.0.0.1:5000/ to view
tables and plots directly from the CSVs in this folder.

Profiler templates:
- CPU: `sbatch slurm/profile_cpu.sbatch`
- GPU: `sbatch slurm/profile_gpu.sbatch`
