extern "C" {

#include <cuda.h>
#include <cstdio>
#include "timer.h"
#include "calcpi.h"

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

struct calc_result_t do_calcpi(int worksize, int iterations) {
    calc_result_t endResult;
    init_end_result(&endResult);

    starttimer(&endResult.total_time);

    starttimer(&endResult.alloc_time);
#define BLOCK_SIZE 256
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

    stoptimer(&endResult.alloc_time);
    starttimer(&endResult.calc_time);
    double mypi = 0.0;

    for (int i = 0; i < iterations; i += BATCH_SIZE) {
        cudaMemset(device_pieparts, 0, sizeof(double) * BATCH_SIZE);
        int actualSize = BATCH_SIZE;
        if (actualSize > (iterations - i)) {
            actualSize = iterations - i;
        }
        cudaMemcpy(device_offset, &i, sizeof(int), cudaMemcpyHostToDevice);
        int blocks = (BATCH_SIZE / BLOCK_SIZE);
        pi_iter<<<blocks, BLOCK_SIZE>>>(device_offset, device_iterations, device_m, device_pieparts);
        int offset = 1;
        do {
            add<<<blocks, BLOCK_SIZE>>>(device_pieparts, offset, actualSize);
            offset *= 2;
        } while (offset < actualSize);
        double mypi_storage;
        cudaMemcpy(&mypi_storage, device_pieparts, sizeof(double), cudaMemcpyDeviceToHost);

        mypi += mypi_storage * m;
    }

    stoptimer(&endResult.calc_time);
    starttimer(&endResult.dealloc_time);
    cudaFree(device_pieparts);
    cudaFree(device_iterations);
    cudaFree(device_m);
    cudaFree(device_offset);
    free(host_pieparts);
    stoptimer(&endResult.dealloc_time);

    return endResult;
}

struct calc_result_t calc_pi(int worksize, int iterations) {
    return do_calcpi(worksize, iterations);
}

};