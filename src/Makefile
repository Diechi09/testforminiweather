# Makefile for building the miniWeather-like MPI+OpenMP mini-app.
# Uses mpicxx by default; override CXX from environment if needed.

CXX ?= mpicxx
CXXFLAGS ?= -O3 -fopenmp -std=c++11
LDFLAGS ?= -fopenmp
TARGET = miniweather_mpi_omp
SRC = miniweather_mpi_omp.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

.PHONY: all clean
