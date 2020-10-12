extern "C" {

#include <cuda.h>
#include <cstdio>
#include "timer.h"
#include "calcpi.h"

__global__ void pi_iter(const unsigned int count, const unsigned int max, const double *m, double *pieparts) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    for (unsigned int i = count * index; i < count * (index + 1) && i < max; i++) {
        double n_i = ((double) i + 0.5) * *m;
        pieparts[index] += 4.0 / (1.0 + n_i * n_i);
    }
    pieparts[index] *= *m;
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
#define GRIDS 1024
#define BLOCKS 140

    starttimer(&endResult.alloc_time);
    int batch_size = GRIDS * BLOCKS;

    unsigned int perThread = (iterations + batch_size) / batch_size;

    int *device_offset;
    cudaMalloc(&device_offset, sizeof(int));

    double *device_pieparts;
    cudaMalloc(&device_pieparts, sizeof(double) * batch_size);

    double *host_pieparts = (double *) malloc(sizeof(double) * batch_size);

    double m = 1.0 / (double) iterations;

    double *device_m;
    cudaMalloc(&device_m, sizeof(double));
    cudaMemcpy(device_m, &m, sizeof(double), cudaMemcpyHostToDevice);

    stoptimer(&endResult.alloc_time);
    starttimer(&endResult.calc_time);

    pi_iter<<<GRIDS, BLOCKS>>>(perThread, iterations, device_m, device_pieparts);
    cudaMemcpy(host_pieparts, device_pieparts, sizeof(double) * batch_size, cudaMemcpyDeviceToHost);
    double mypi;
    for (int i = 0; i < batch_size; i++) {
        mypi += host_pieparts[i];
    }
    stoptimer(&endResult.calc_time);
    starttimer(&endResult.dealloc_time);
    cudaFree(device_pieparts);
    cudaFree(device_m);
    cudaFree(device_offset);
    free(host_pieparts);
    stoptimer(&endResult.dealloc_time);
    stoptimer(&endResult.total_time);
    endResult.pi_value = mypi;

    return endResult;
}

struct calc_result_t calc_pi(int worksize, int iterations) {
    return do_calcpi(worksize, iterations);
}

}
