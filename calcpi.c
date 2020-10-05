//
// Created by user on 2020-10-05.
//
#include "calcpi.h"

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

