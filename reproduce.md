# Reproduce runs on Magic Castle / Alliance clusters (CPU or GPU)

1. **Clone repo**
   ```bash
   git clone <repo_url>
   cd testforminiweather
   ```

2. **Load modules** (adapt names/versions)
   ```bash
   source env/load_modules.sh
   ```

3. **Build**
   - CPU: `cd src && make cpu`
   - GPU: `cd src && make gpu`

4. **Run a 1-node baseline**
   - CPU: `srun -n 1 ./miniweather_mpi_omp --nx 256 --nz 128 --steps 50 --output ../results/cpu_baseline.csv`
   - GPU: `srun -n 1 --gpus-per-task=1 ./miniweather_mpi_cuda --nx 256 --nz 128 --steps 50 --output ../results/gpu_baseline.csv --gpu`

5. **Strong-scaling (GPU)**
   - Submit multiple jobs varying nodes/GPUs:
     ```bash
     sbatch slurm/miniweather_strong.sbatch
     ```
   - Append rows from `results/strong_<jobid>.csv` into `results/strong_scaling.csv`.

6. **Weak-scaling (GPU)**
   - Submit jobs with different `-N` to grow total grid size:
     ```bash
     sbatch slurm/miniweather_weak.sbatch
     ```
   - Append rows into `results/weak_scaling.csv`.

7. **Profiling**
   - CPU example with perf/likwid fallback:
     ```bash
     sbatch slurm/profile_cpu.sbatch
     ```
   - GPU example with Nsight Systems fallback to plain run:
     ```bash
     sbatch slurm/profile_gpu.sbatch
     ```
   - Store profiler outputs in `results/`.

8. **Plot scaling**
   ```bash
   python src/plot_scaling.py
   ```

9. **Record system info**
   - Fill `SYSTEM.md` with exact node/CPU/GPU/interconnect/module details.

10. **Container option**
    - Build the Apptainer image: `apptainer build env/project.sif env/project.def`
    - Run inside container (GPU):
      ```bash
      ./slurm/run.sh apptainer exec --nv --bind $PWD:$PWD --pwd $PWD env/project.sif \
        srun --gpus-per-task=1 ./miniweather_mpi_cuda --nx 256 --nz 128 --steps 50 --output results/gpu_container.csv --gpu
      ```

11. **Local frontend (no Slurm required)**
    - Visualise CSVs and quick plots on localhost:
      ```bash
      python -m venv .venv && source .venv/bin/activate
      pip install flask matplotlib pandas
      python src/frontend.py  # browse http://127.0.0.1:5000/
      ```
