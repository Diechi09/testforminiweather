# Reproducing Experiments

Follow these steps to rebuild and rerun the miniWeather mini-app on Magic Castle
or similar Alliance clusters.

1. **Clone the repository**
```bash
git clone <repo-url>
cd testforminiweather
```

2. **Load modules or build container**
- Option A: modules
```bash
source env/load_modules.sh  # edit module versions to match your system
```
- Option B: Apptainer (optional)
```bash
apptainer build env/project.sif env/project.def
# Then prefix commands with: ./slurm/run.sh <command>
```

3. **Build the code**
```bash
cd src
make
cd ..
```

4. **Run a 1-node baseline** (example with 2 ranks, 4 threads each)
```bash
srun -n 2 --cpus-per-task=4 ./miniweather_mpi_omp --nx 256 --nz 128 --steps 50 --output results/baseline.csv
```

5. **Strong scaling campaign**
- Submit `slurm/miniweather_strong.sbatch` with varying `-N` values.
- After runs finish, concatenate outputs into `results/strong_scaling.csv`.

6. **Weak scaling campaign**
- Submit `slurm/miniweather_weak.sbatch` with varying `-N` values.
- Aggregate outputs into `results/weak_scaling.csv`.

7. **Profiling**
- Use `slurm/profile_cpu.sbatch` to collect perf/likwid statistics for a single
  representative run. Save logs in `results/`.

8. **Plot scaling results**
```bash
python src/plot_scaling.py
```
This generates `strong_scaling.(png|svg)` and `weak_scaling.(png|svg)` under
`results/`.

9. **Document system details**
- Fill in `SYSTEM.md` with node architecture, software versions, and environment
  variables used during runs.

TODOs:
- Replace `<repo-url>` and `<account>` placeholders with real values.
- Record exact module versions in env/modules.txt for reproducibility.
