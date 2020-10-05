//
// Created by user on 2020-09-28.
//

#ifndef INC_2DV605_CUDA_CALCPI_H
#define INC_2DV605_CUDA_CALCPI_H

#include "timer.h"

typedef struct calc_result_t {
    double pi_value;
    timekeeper_t total_time;
    timekeeper_t alloc_time;
    timekeeper_t dealloc_time;
    timekeeper_t calc_time;
} calc_result_t;

struct calc_result_t calc_pi(int worksize, int iterations);

void init_end_result(calc_result_t *result) {
    result->total_time.seconds = 0;
    result->total_time.nanos = 0;
    result->alloc_time.seconds = 0;
    result->alloc_time.nanos = 0;
    result->dealloc_time.seconds = 0;
    result->dealloc_time.nanos = 0;
    result->calc_time.seconds = 0;
    result->calc_time.nanos = 0;
}

#endif //INC_2DV605_CUDA_CALCPI_H
