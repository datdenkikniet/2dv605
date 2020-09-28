#include <cstdio>
#include <cuda.h>

__global__ void pi_iter(int iterations, double m, double *pieparts) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < iterations) {
        double n_i = ((double) index + 0.5) * m;
        pieparts[index] = 4.0 / (1.0 + n_i * n_i);
    }
}

// TODO No worky if > 48000000 iterations
// Add extra timers to determine actual computation time (and exclude malloc time)
double do_calcpi(int iterations) {
    double *device_pieparts;
    cudaMalloc(&device_pieparts, sizeof(double) * iterations);

    int total_cells = iterations;

    int *device_iterations;
    cudaMalloc(&device_iterations, sizeof(int));
    cudaMemcpy(&iterations, device_iterations, sizeof(int), cudaMemcpyHostToDevice);

    int *device_indices;
    cudaMalloc(&device_indices, sizeof(int) * total_cells);

    double *host_pieparts = (double *) malloc(sizeof(double) * total_cells);

    for (int i = 0; i < total_cells; i++) {
        host_pieparts[i] = 0.0;
    }

    double m = 1.0 / (double) iterations;

    int blocks = (iterations / 1024) + 1;
    pi_iter<<<blocks, 1024>>>(iterations, m, device_pieparts);

    cudaMemcpy(host_pieparts, device_pieparts, sizeof(double) * total_cells, cudaMemcpyDeviceToHost);

    double mypi = 0.0;

    for (int i = 0; i < total_cells; i++){
        mypi += host_pieparts[i];
    }

    mypi *= m;

    cudaFree(device_pieparts);
    free(host_pieparts);

    return mypi;
}

extern "C" {

double calc_pi(int iterations) {
    return do_calcpi(iterations);
}

};