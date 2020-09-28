#include <cstdio>
#include <cuda.h>

double do_calcpi(int iterations) {
    cudaDeviceProp deviceProp = {};
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("%d\n", deviceProp.maxThreadsPerBlock);
    return 3.14;
}

extern "C" {

double calc_pi(int iterations) {

    return do_calcpi(iterations);
}

};

