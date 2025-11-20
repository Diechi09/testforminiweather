# results/

Store experiment outputs here.
- Strong scaling CSV: `results/strong_scaling.csv` (aggregate from multiple strong-scaling job outputs).
- Weak scaling CSV: `results/weak_scaling.csv` (aggregate from weak-scaling job outputs).
- Profiling logs: `results/profile_cpu_<details>.txt`, `results/nsys_<jobid>.qdrep`, etc. for GPU profiling.

Run `python src/plot_scaling.py` after collecting data to generate
`strong_scaling.(png|svg)` and `weak_scaling.(png|svg)` in this folder.
