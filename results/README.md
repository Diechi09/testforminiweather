# results/

Store experiment outputs here.
- Strong scaling CSV: `results/strong_scaling.csv` (aggregate from multiple strong-scaling job outputs).
- Weak scaling CSV: `results/weak_scaling.csv` (aggregate from weak-scaling job outputs).
- Profiling logs: `results/profile_cpu_<details>.txt`, `results/nsys_<jobid>.qdrep`, etc. for GPU profiling.

Run `python src/plot_scaling.py` after collecting data to generate
`strong_scaling.(png|svg)` and `weak_scaling.(png|svg)` in this folder.

For local demos without Slurm, launch the Flask frontend and open the default
address: `python src/frontend.py` then browse http://127.0.0.1:5000/ to view
tables and plots directly from the CSVs in this folder.

Profiler templates:
- CPU: `sbatch slurm/profile_cpu.sbatch`
- GPU: `sbatch slurm/profile_gpu.sbatch`
