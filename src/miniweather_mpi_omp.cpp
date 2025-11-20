// miniweather_mpi_omp.cpp
// Hybrid MPI + OpenMP miniWeather-like 2D stencil mini-app.
// Implements a simple advection-diffusion update on a 2D grid with halo exchange.
// Designed for course assignments on Magic Castle/Alliance clusters using Slurm.

#include <mpi.h>
#include <omp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>

struct Domain {
    int global_nx;
    int global_nz;
    int local_nx; // includes interior cells only (no halos)
    int nz;
    int halo; // number of halo layers on each side
    int rank, size;
    int left, right;
    double dx, dz, dt;
    std::vector<double> field;      // current state including halos
    std::vector<double> field_next; // updated state including halos
};

struct Options {
    int nx = 256;
    int nz = 128;
    int steps = 100;
    std::string output = "results/run_summary.csv";
};

// Parse simple command-line options; minimal error checking for clarity.
Options parse_args(int argc, char **argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--nx") == 0 && i + 1 < argc) {
            opt.nx = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--nz") == 0 && i + 1 < argc) {
            opt.nz = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            opt.steps = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            opt.output = argv[++i];
        } else if (strcmp(argv[i], "--help") == 0) {
            if (MPI::COMM_WORLD.Get_rank() == 0) {
                std::cout << "Usage: ./miniweather_mpi_omp [--nx N] [--nz N] [--steps N] [--output path]\n";
            }
            MPI_Finalize();
            std::exit(0);
        }
    }
    return opt;
}

// Initialize domain decomposition and arrays.
void setup_domain(Domain &dom, const Options &opt) {
    dom.global_nx = opt.nx;
    dom.global_nz = opt.nz;
    dom.nz = opt.nz;
    dom.halo = 1; // 5-point stencil needs one layer on each side

    dom.rank = 0;
    dom.size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &dom.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &dom.size);

    dom.left = (dom.rank == 0) ? MPI_PROC_NULL : dom.rank - 1;
    dom.right = (dom.rank == dom.size - 1) ? MPI_PROC_NULL : dom.rank + 1;

    // Split x-dimension among ranks as evenly as possible.
    int base_nx = dom.global_nx / dom.size;
    int remainder = dom.global_nx % dom.size;
    dom.local_nx = base_nx + (dom.rank < remainder ? 1 : 0);

    // Spatial step sizes (units arbitrary); CFL-friendly dt.
    dom.dx = 1.0;
    dom.dz = 1.0;
    double c = 1.0; // advection speed
    double diffusion = 0.1;
    dom.dt = 0.4 * dom.dx / (c + 4.0 * diffusion / dom.dx);

    // Allocate arrays with halo layers on each side in x.
    size_t total_cells = (dom.local_nx + 2 * dom.halo) * dom.nz;
    dom.field.assign(total_cells, 0.0);
    dom.field_next.assign(total_cells, 0.0);
}

// Access helper to flatten 2D index (i: x, k: z) including halos.
inline size_t idx(const Domain &dom, int i, int k) {
    int nx_with_halo = dom.local_nx + 2 * dom.halo;
    return static_cast<size_t>(k) * nx_with_halo + i;
}

// Initialize a "warm bubble" in the center of the global domain.
void initialize_field(Domain &dom) {
    int global_start = 0;
    // Compute global start index for this rank based on decomposition.
    int base_nx = dom.global_nx / dom.size;
    int remainder = dom.global_nx % dom.size;
    for (int r = 0; r < dom.rank; ++r) {
        global_start += base_nx + (r < remainder ? 1 : 0);
    }

    double cx = dom.global_nx / 2.0;
    double cz = dom.global_nz / 2.0;

    for (int k = 0; k < dom.nz; ++k) {
        for (int i = 0; i < dom.local_nx; ++i) {
            double gx = global_start + i;
            double dist2 = (gx - cx) * (gx - cx) + (k - cz) * (k - cz);
            double value = std::exp(-dist2 / (0.1 * dom.global_nx * dom.global_nz));
            dom.field[idx(dom, i + dom.halo, k)] = value;
        }
    }
}

// Exchange halo regions with neighbors in x-direction.
void exchange_halos(Domain &dom) {
    int nx_with_halo = dom.local_nx + 2 * dom.halo;
    MPI_Request reqs[4];
    // Send to left, receive from right
    MPI_Isend(&dom.field[idx(dom, dom.halo, 0)], dom.nz, MPI_DOUBLE, dom.left, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&dom.field[idx(dom, dom.local_nx + dom.halo, 0)], dom.nz, MPI_DOUBLE, dom.right, 0, MPI_COMM_WORLD, &reqs[1]);
    // Send to right, receive from left
    MPI_Isend(&dom.field[idx(dom, dom.local_nx - 1 + dom.halo, 0)], dom.nz, MPI_DOUBLE, dom.right, 1, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(&dom.field[idx(dom, 0, 0)], dom.nz, MPI_DOUBLE, dom.left, 1, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
}

// Perform one advection-diffusion update using a 5-point stencil.
void update(Domain &dom) {
    double c = 1.0;
    double diffusion = 0.1;

    #pragma omp parallel for collapse(2)
    for (int k = 1; k < dom.nz - 1; ++k) {
        for (int i = dom.halo; i < dom.local_nx + dom.halo; ++i) {
            double center = dom.field[idx(dom, i, k)];
            double left = dom.field[idx(dom, i - 1, k)];
            double right = dom.field[idx(dom, i + 1, k)];
            double down = dom.field[idx(dom, i, k - 1)];
            double up = dom.field[idx(dom, i, k + 1)];

            double advect_x = -c * (right - left) / (2.0 * dom.dx);
            double advect_z = -c * (up - down) / (2.0 * dom.dz);
            double laplacian = (left + right + up + down - 4.0 * center) / (dom.dx * dom.dx);

            dom.field_next[idx(dom, i, k)] = center + dom.dt * (advect_x + advect_z + diffusion * laplacian);
        }
    }

    // Swap arrays for next iteration.
    dom.field.swap(dom.field_next);
}

// Compute global sum for diagnostics.
double compute_global_sum(const Domain &dom) {
    double local_sum = 0.0;
    for (int k = 1; k < dom.nz - 1; ++k) {
        for (int i = dom.halo; i < dom.local_nx + dom.halo; ++i) {
            local_sum += dom.field[idx(dom, i, k)];
        }
    }
    double global_sum = 0.0;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return global_sum;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    Options opt = parse_args(argc, argv);

    Domain dom;
    setup_domain(dom, opt);
    initialize_field(dom);

    double start = MPI_Wtime();
    for (int step = 0; step < opt.steps; ++step) {
        exchange_halos(dom);
        update(dom);
        if (step % 10 == 0) {
            double sum = compute_global_sum(dom);
            if (dom.rank == 0) {
                std::printf("Step %d global sum = %e\n", step, sum);
            }
        }
    }
    double end = MPI_Wtime();

    double local_time = end - start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int provided = 0;
    MPI_Query_thread(&provided);

    if (dom.rank == 0) {
        int omp_threads = omp_get_max_threads();
        FILE *f = std::fopen(opt.output.c_str(), "a");
        if (f) {
            std::fprintf(f, "ranks,threads,nx,nz,steps,total_time,time_per_step,mpi_thread_level\n");
            std::fprintf(f, "%d,%d,%d,%d,%d,%f,%f,%d\n", dom.size, omp_threads, opt.nx, opt.nz, opt.steps, max_time, max_time / opt.steps, provided);
            std::fclose(f);
        } else {
            std::perror("Failed to open output file");
        }
        std::printf("Run complete: %d ranks, %d threads, time %.3f s\n", dom.size, omp_threads, max_time);
    }

    MPI_Finalize();
    return 0;
}
