extern "C" {

#include <cuda.h>
#include <cstdio>
#include "timer.h"

__global__ void pi_iter(const int *offset, const int *iterations, const double *m, double *pieparts) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x + *offset;
    if (index < *iterations) {
        double n_i = ((double) index + 0.5) * *m;
        pieparts[index - *offset] = 4.0 / (1.0 + n_i * n_i);
    }
}

__global__ void add(double *values, const int offset, const int max) {
    unsigned int index = (blockIdx.x * blockDim.x + threadIdx.x) * offset * 2;
    if (index + offset < max) {
        values[index] = values[index] + values[index + offset];
        values[index + offset] = 0;
    }
}

// Add extra timers to determine actual computation time (and exclude malloc time)
double do_calcpi(int worksize, int iterations) {
    int BATCH_SIZE = worksize;
    int *device_offset;
    cudaMalloc(&device_offset, sizeof(int));

    int *device_iterations;
    cudaMalloc(&device_iterations, sizeof(int));
    cudaMemcpy(&iterations, device_iterations, sizeof(int), cudaMemcpyHostToDevice);

    double *device_pieparts;
    cudaMalloc(&device_pieparts, sizeof(double) * BATCH_SIZE);

    double *host_pieparts = (double *) malloc(sizeof(double) * BATCH_SIZE);

    double m = 1.0 / (double) iterations;

    double *device_m;
    cudaMalloc(&device_m, sizeof(double));
    cudaMemcpy(device_m, &m, sizeof(double), cudaMemcpyHostToDevice);

    double mypi = 0.0;

    timekeeper_t calctime;
    starttimer(&calctime);
    for (int i = 0; i < iterations; i += BATCH_SIZE) {
        int actualSize = BATCH_SIZE;
        if (actualSize > (iterations - i)) {
            actualSize = iterations - i;
        }
        cudaMemcpy(device_offset, &i, sizeof(int), cudaMemcpyHostToDevice);
        int blocks = (BATCH_SIZE / 1024) + 1;
        pi_iter<<<blocks, 1024>>>(device_offset, device_iterations, device_m, device_pieparts);
        int offset = 1;
        do {
            add<<<blocks, 1024>>>(device_pieparts, offset, actualSize);
            offset *= 2;
        } while (offset < actualSize);

        double mypi_storage;
        cudaMemcpy(&mypi_storage, device_pieparts, sizeof(double), cudaMemcpyDeviceToHost);

        mypi += mypi_storage * m;
    }

    stoptimer(&calctime);
    printf("%li.%03li\n", calctime.seconds, calctime.nanos/1000000);
    cudaFree(device_pieparts);
    free(host_pieparts);

    return mypi;
}
double calc_pi(int worksize, int iterations) {
    return do_calcpi(worksize, iterations);
}

};