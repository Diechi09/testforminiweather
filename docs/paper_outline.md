# Short Paper Outline (4–6 pages)

## Introduction & Problem Statement
- Motivation for miniWeather-style 2D stencil as HPC teaching app.
- Relevance to atmospheric modeling and simplified physics.
- Objectives of the assignment (scaling, profiling, hybrid MPI+OpenMP).

## Methodology
- Code structure: key files, build system, and dependencies.
- Domain decomposition in x with halo cells; OpenMP over local subdomain.
- Numerical scheme: advection-diffusion, 5-point stencil, CFL considerations.

## Experimental Setup
- Cluster description (fill from SYSTEM.md: CPUs, memory, interconnect).
- Software stack: modules used, compiler/MPI versions.
- Problem sizes, time steps, and mapping to cores/nodes for strong/weak scaling.

## Results
- Strong scaling: table/plots of speedup and efficiency.
- Weak scaling: time per step vs nodes and discussion of scaling limits.
- Include representative wall-clock times and MPI/OpenMP breakdown if available.

## Profiling & Bottleneck Analysis
- Tools used (perf/likwid). Key metrics (IPC, bandwidth, cache misses).
- Hot loops (stencil update) and communication overhead (halo exchange).

## Optimisation
- Changes applied (e.g., better layout, OpenMP schedule, message aggregation).
- Impact on performance and efficiency; before/after comparison.

## Limitations and Future Work
- Numerical simplifications vs real weather codes.
- Potential GPU port or vectorisation.
- Additional physics or boundary conditions.
