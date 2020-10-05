GCC = gcc -Wall
GCCOMP = $(GCC) -fopenmp -DCOMPILE_OPENMP
GCCNL = $(GCC) -c
GCCNLOMP = $(GCCNL) -fopenmp -DCOMPILE_OPENMP

NVCC = nvcc -DCOMPILE_CUDA
NVCCNL = $(NVCC) -c

all: calcpi-openmp calcpi-cuda

calcpi-cuda: cuda-main.o timer.o calcpi-cuda.o calcpi.o
	$(NVCC) -o calcpi-cuda cuda-main.o timer.o calcpi-cuda.o calcpi.o

calcpi-openmp: openmp-main.o timer.o calcpi-openmp.o calcpi.o
	$(GCCOMP) -o calcpi-openmp openmp-main.o timer.o calcpi-openmp.o calcpi.o

cuda-main.o: main.c
	$(NVCCNL) main.c -o cuda-main.o

openmp-main.o: main.c
	$(GCCNLOMP) main.c -o openmp-main.o

calcpi.o: calcpi.c
	$(GCCNL) calcpi.c -o calcpi.o

timer.o: timer.c
	$(GCCNL) timer.c -o timer.o

calcpi-cuda.o: calcpi-cuda.cu
	$(NVCCNL) calcpi-cuda.cu -o calcpi-cuda.o

calcpi-openmp.o: calcpi-openmp.c
	$(GCCNLOMP) calcpi-openmp.c -o calcpi-openmp.o

clean:
	rm -rf *.o
	rm -f calcpi-openmp
	rm -f calcpi-cuda