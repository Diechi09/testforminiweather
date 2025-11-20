// miniweather_mpi_omp.cpp
// Hybrid MPI miniWeather-like 2D stencil mini-app with optional CUDA acceleration.
// Implements a simple advection-diffusion update on a 2D grid with halo exchange.
// Designed for coursework on Magic Castle/Alliance clusters using Slurm and supports
// running on CPU (MPI+OpenMP) or GPU (MPI+CUDA) with one GPU per MPI rank.

#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

struct Domain {
    int global_nx;
    int global_nz;
    int local_nx; // interior cells only (no halos)
    int nz;
    int halo;     // number of halo layers on each side
    int rank, size;
    int left, right;
    double dx, dz, dt;
    std::vector<double> field;      // host current state including halos
    std::vector<double> field_next; // host updated state including halos

#ifdef USE_CUDA
    double *d_field = nullptr;
    double *d_field_next = nullptr;
    std::vector<double> h_left_send, h_right_send, h_left_recv, h_right_recv; // halo buffers
#endif
};

struct Options {
    int nx = 256;
    int nz = 128;
    int steps = 100;
    std::string output = "results/run_summary.csv";
    bool use_gpu =
#ifdef USE_CUDA
        true;
#else
        false;
#endif
};

#ifdef USE_CUDA
#define CUDA_CALL(call)                                                                 \
    do {                                                                                \
        cudaError_t err = (call);                                                       \
        if (err != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), \
                         __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                               \
        }                                                                               \
    } while (0)
#endif

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
        } else if (strcmp(argv[i], "--gpu") == 0) {
            opt.use_gpu = true;
        } else if (strcmp(argv[i], "--cpu") == 0) {
            opt.use_gpu = false;
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "Usage: ./miniweather_mpi_omp [--nx N] [--nz N] [--steps N] [--output path] [--gpu|--cpu]\n";
            MPI_Finalize();
            std::exit(0);
        }
    }
#ifndef USE_CUDA
    opt.use_gpu = false; // ensure CPU path if compiled without CUDA
#endif
    return opt;
}

// Access helper to flatten 2D index (i: x, k: z) including halos.
inline size_t idx(const Domain &dom, int i, int k) {
    int nx_with_halo = dom.local_nx + 2 * dom.halo;
    return static_cast<size_t>(k) * nx_with_halo + i;
}

// Utility to check whether an output file already has content to avoid
// repeatedly writing CSV headers during multi-run experiments.
bool file_has_content(const std::string &path) {
    struct stat st {};
    return (stat(path.c_str(), &st) == 0) && st.st_size > 0;
}

// Initialize domain decomposition and arrays.
void setup_domain(Domain &dom, const Options &opt) {
    dom.global_nx = opt.nx;
    dom.global_nz = opt.nz;
    dom.nz = opt.nz;
    dom.halo = 1; // 5-point stencil needs one layer on each side

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
    double c = 1.0;      // advection speed
    double diffusion = 0.1;
    dom.dt = 0.4 * dom.dx / (c + 4.0 * diffusion / dom.dx);

    // Allocate arrays with halo layers on each side in x.
    size_t total_cells = (dom.local_nx + 2 * dom.halo) * dom.nz;
    dom.field.assign(total_cells, 0.0);
    dom.field_next.assign(total_cells, 0.0);
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

#ifdef USE_CUDA
// Allocate device buffers and halo staging buffers.
void setup_device(Domain &dom) {
    size_t total_cells = (dom.local_nx + 2 * dom.halo) * dom.nz;
    CUDA_CALL(cudaMalloc(&dom.d_field, total_cells * sizeof(double)));
    CUDA_CALL(cudaMalloc(&dom.d_field_next, total_cells * sizeof(double)));
    // Staging halos for MPI communication (host memory).
    dom.h_left_send.resize(dom.nz, 0.0);
    dom.h_right_send.resize(dom.nz, 0.0);
    dom.h_left_recv.resize(dom.nz, 0.0);
    dom.h_right_recv.resize(dom.nz, 0.0);
    CUDA_CALL(cudaMemcpy(dom.d_field, dom.field.data(), total_cells * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dom.d_field_next, dom.field_next.data(), total_cells * sizeof(double), cudaMemcpyHostToDevice));
}

// Copy halo columns from device to host buffers for MPI exchange.
void pack_device_halos(Domain &dom) {
    size_t nx_pitch = static_cast<size_t>(dom.local_nx + 2 * dom.halo) * sizeof(double);
    int left_col = dom.halo;
    int right_col = dom.local_nx + dom.halo - 1;
    CUDA_CALL(cudaMemcpy2D(dom.h_left_send.data(), sizeof(double),
                           dom.d_field + left_col, nx_pitch,
                           sizeof(double), dom.nz, cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy2D(dom.h_right_send.data(), sizeof(double),
                           dom.d_field + right_col, nx_pitch,
                           sizeof(double), dom.nz, cudaMemcpyDeviceToHost));
}

// Scatter received halo data from host buffers into device halos.
void unpack_device_halos(Domain &dom) {
    size_t nx_pitch = static_cast<size_t>(dom.local_nx + 2 * dom.halo) * sizeof(double);
    CUDA_CALL(cudaMemcpy2D(dom.d_field + dom.local_nx + dom.halo, nx_pitch,
                           dom.h_right_recv.data(), sizeof(double),
                           sizeof(double), dom.nz, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy2D(dom.d_field + 0, nx_pitch,
                           dom.h_left_recv.data(), sizeof(double),
                           sizeof(double), dom.nz, cudaMemcpyHostToDevice));
}

// CUDA kernel performing one advection-diffusion update.
__global__ void update_kernel(double *out, const double *in, int nx, int nz, int halo,
                              double dx, double dz, double dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + halo; // skip left halo
    int k = blockIdx.y * blockDim.y + threadIdx.y + 1;    // skip bottom boundary
    if (i >= halo && i < nx + halo && k >= 1 && k < nz - 1) {
        int nx_with_halo = nx + 2 * halo;
        size_t idx_center = static_cast<size_t>(k) * nx_with_halo + i;
        double center = in[idx_center];
        double left = in[idx_center - 1];
        double right = in[idx_center + 1];
        double down = in[idx_center - nx_with_halo];
        double up = in[idx_center + nx_with_halo];
        double c = 1.0;
        double diffusion = 0.1;
        double advect_x = -c * (right - left) / (2.0 * dx);
        double advect_z = -c * (up - down) / (2.0 * dz);
        double laplacian = (left + right + up + down - 4.0 * center) / (dx * dx);
        out[idx_center] = center + dt * (advect_x + advect_z + diffusion * laplacian);
    }
}
#endif

// Exchange halo regions with neighbors in x-direction (CPU data).
void exchange_halos_cpu(Domain &dom) {
    MPI_Request reqs[4];
    MPI_Isend(&dom.field[idx(dom, dom.halo, 0)], dom.nz, MPI_DOUBLE, dom.left, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(&dom.field[idx(dom, dom.local_nx + dom.halo, 0)], dom.nz, MPI_DOUBLE, dom.right, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(&dom.field[idx(dom, dom.local_nx - 1 + dom.halo, 0)], dom.nz, MPI_DOUBLE, dom.right, 1, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(&dom.field[idx(dom, 0, 0)], dom.nz, MPI_DOUBLE, dom.left, 1, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
}

#ifdef USE_CUDA
// Exchange halos for GPU data using host staging buffers.
void exchange_halos_gpu(Domain &dom) {
    pack_device_halos(dom);
    MPI_Request reqs[4];
    MPI_Isend(dom.h_left_send.data(), dom.nz, MPI_DOUBLE, dom.left, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(dom.h_right_recv.data(), dom.nz, MPI_DOUBLE, dom.right, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Isend(dom.h_right_send.data(), dom.nz, MPI_DOUBLE, dom.right, 1, MPI_COMM_WORLD, &reqs[2]);
    MPI_Irecv(dom.h_left_recv.data(), dom.nz, MPI_DOUBLE, dom.left, 1, MPI_COMM_WORLD, &reqs[3]);
    MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);
    unpack_device_halos(dom);
}
#endif

// Perform one advection-diffusion update on CPU using OpenMP.
void update_cpu(Domain &dom) {
    double c = 1.0;
    double diffusion = 0.1;
#ifdef _OPENMP
    #pragma omp parallel for collapse(2)
#endif
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
    dom.field.swap(dom.field_next);
}

#ifdef USE_CUDA
void update_gpu(Domain &dom) {
    dim3 block(16, 8);
    dim3 grid((dom.local_nx + block.x - 1) / block.x, (dom.nz - 2 + block.y - 1) / block.y);
    update_kernel<<<grid, block>>>(dom.d_field_next, dom.d_field, dom.local_nx, dom.nz, dom.halo, dom.dx, dom.dz, dom.dt);
    CUDA_CALL(cudaGetLastError());
    CUDA_CALL(cudaDeviceSynchronize());
    std::swap(dom.d_field, dom.d_field_next);
}

// Copy device field to host for diagnostics.
void copy_device_to_host(Domain &dom) {
    size_t total_cells = (dom.local_nx + 2 * dom.halo) * dom.nz;
    CUDA_CALL(cudaMemcpy(dom.field.data(), dom.d_field, total_cells * sizeof(double), cudaMemcpyDeviceToHost));
}
#endif

// Compute global sum for diagnostics.
double compute_global_sum(Domain &dom, bool use_gpu) {
#ifdef USE_CUDA
    if (use_gpu) {
        copy_device_to_host(dom);
    }
#endif
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

#ifdef USE_CUDA
    if (opt.use_gpu) {
        int device_count = 0;
        CUDA_CALL(cudaGetDeviceCount(&device_count));
        if (device_count == 0) {
            if (dom.rank == 0) {
                std::fprintf(stderr, "No CUDA devices detected; falling back to CPU path.\n");
            }
            opt.use_gpu = false;
        } else {
            int device_id = dom.rank % device_count;
            CUDA_CALL(cudaSetDevice(device_id));
            setup_device(dom);
        }
    }
#endif

    double start = MPI_Wtime();
    double comm_time = 0.0;
    double compute_time = 0.0;
    for (int step = 0; step < opt.steps; ++step) {
#ifdef USE_CUDA
        if (opt.use_gpu) {
            double comm_start = MPI_Wtime();
            exchange_halos_gpu(dom);
            double comm_end = MPI_Wtime();
            comm_time += comm_end - comm_start;

            double compute_start = MPI_Wtime();
            update_gpu(dom);
            double compute_end = MPI_Wtime();
            compute_time += compute_end - compute_start;
        } else
#endif
        {
            double comm_start = MPI_Wtime();
            exchange_halos_cpu(dom);
            double comm_end = MPI_Wtime();
            comm_time += comm_end - comm_start;

            double compute_start = MPI_Wtime();
            update_cpu(dom);
            double compute_end = MPI_Wtime();
            compute_time += compute_end - compute_start;
        }
        if (step % 10 == 0) {
            double sum = compute_global_sum(dom, opt.use_gpu);
            if (dom.rank == 0) {
                std::printf("Step %d global sum = %e\n", step, sum);
            }
        }
    }
    double end = MPI_Wtime();

    double local_time = end - start;
    double max_time = 0.0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double max_comm = 0.0;
    double max_compute = 0.0;
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&compute_time, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int provided = 0;
    MPI_Query_thread(&provided);

    if (dom.rank == 0) {
#ifdef _OPENMP
        int omp_threads = omp_get_max_threads();
#else
        int omp_threads = 1;
#endif
        const char *nodes_env = std::getenv("SLURM_JOB_NUM_NODES");
        int nodes = nodes_env ? std::atoi(nodes_env) : 1;
        double avg_comm = max_comm / opt.steps;
        double avg_compute = max_compute / opt.steps;
        double time_per_step = max_time / opt.steps;
        double comm_fraction = (time_per_step > 0.0) ? avg_comm / time_per_step : 0.0;
        double cells = static_cast<double>(opt.nx) * static_cast<double>(opt.nz) * static_cast<double>(opt.steps);
        double throughput_mcells = (max_time > 0.0) ? (cells / max_time) / 1.0e6 : 0.0;

        bool header_needed = !file_has_content(opt.output);
        FILE *f = std::fopen(opt.output.c_str(), "a");
        if (f) {
            if (header_needed) {
                std::fprintf(f, "nodes,ranks,threads,nx,nz,steps,total_time,time_per_step,avg_comm_per_step,avg_compute_per_step,comm_fraction,throughput_mcells_s,mpi_thread_level,use_gpu\n");
            }
            std::fprintf(f, "%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%d,%d\n", nodes, dom.size, omp_threads, opt.nx, opt.nz, opt.steps, max_time, time_per_step, avg_comm, avg_compute, comm_fraction, throughput_mcells, provided, opt.use_gpu ? 1 : 0);
            std::fclose(f);
        } else {
            std::perror("Failed to open output file");
        }
        std::printf("Run complete: nodes %d, %d ranks, %d threads, gpu %s, time %.3f s (comm %.3e s/step, compute %.3e s/step)\n", nodes, dom.size, omp_threads, opt.use_gpu ? "on" : "off", max_time, avg_comm, avg_compute);
    }

#ifdef USE_CUDA
    if (opt.use_gpu) {
        cudaFree(dom.d_field);
        cudaFree(dom.d_field_next);
    }
#endif

    MPI_Finalize();
    return 0;
}
