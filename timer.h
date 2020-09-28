//
// Created by jona on 2019-11-27.
//

#ifndef TIMER_H
#define TIMER_H

#include <time.h>
#include <stdint.h>

typedef struct {
    struct timespec start;
    struct timespec stop;
    uint64_t seconds;
    uint64_t nanos;
    uint64_t millis;
} timekeeper_t;

void starttimer(timekeeper_t *);
void stoptimer(timekeeper_t *);

#endif //SQLTEST_TIMER_H
