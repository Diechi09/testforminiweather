# Makefile for building the miniWeather-like MPI mini-app.
# Provides CPU (MPI+OpenMP) and GPU (MPI+CUDA) targets.

CXX ?= mpicxx
NVCC ?= nvcc
CXXFLAGS ?= -O3 -fopenmp -std=c++14
NVCCFLAGS ?= -O3 -std=c++14 -Xcompiler -fopenmp
LDFLAGS ?= -fopenmp
TARGET_CPU = miniweather_mpi_omp
TARGET_GPU = miniweather_mpi_cuda
SRC = miniweather_mpi_omp.cpp

all: $(TARGET_CPU)

cpu: $(TARGET_CPU)

gpu: $(TARGET_GPU)

$(TARGET_CPU): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(TARGET_GPU): $(SRC)
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -o $@ $^

clean:
	rm -f $(TARGET_CPU) $(TARGET_GPU) *.o

.PHONY: all clean cpu gpu
