#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <omp.h>
#include <stdlib.h>
#include "timer.h"

#define PI 3.14159265358979323846264

int main(int argc, char *argv[]) {
    int iterations = 192000000;
    int threads = omp_get_max_threads();
    int option;
    while ((option = getopt(argc, argv, "i:t:")) != -1) {
        switch (option) {
            case 'i': {
                char end_char;
                char *end_ptr = &end_char;
                iterations = (int) strtol(optarg, &end_ptr, 10);
                if (iterations < 0 || end_ptr != optarg + strlen(optarg)) {
                    printf("Invalid iteration count\n");
                    return 1;
                }
                break;
            }
            case 't': {
                char end_char;
                char *end_ptr = &end_char;
                threads = (int) strtol(optarg, &end_ptr, 10);
                if (iterations < 0 || end_ptr != optarg + strlen(optarg)) {
                    printf("Invalid thread count\n");
                    return 1;
                }
                if (threads == 0){
                    threads = omp_get_max_threads();
                }
                break;
            }
            default: {
                printf("Unknown option %c\n", option);
                return 1;
            }
        }
    }

    omp_set_num_threads(threads);

    timekeeper_t timer;
    starttimer(&timer);


    double mypi = 0.0;
    double n_i;
    double m = 1.0 / (double) iterations;
    //OMP can intelligently figure out whether the variables should be shared or private, but it is good to write
    //it down explicitly anyways
#pragma omp parallel for reduction(+: mypi) shared(m) private(n_i)
    for (int i = 0; i < iterations; i++) {
        n_i = ((double) i + 0.5) * m;
        mypi += 4.0 / (1.0 + n_i * n_i);
    }

    mypi *= m;
    stoptimer(&timer);
    printf("Computation took %li.%03li seconds\n", timer.seconds, timer.nanos / 1000000);
    printf("     MyPI = %.20lf\n", mypi);
    printf("MyPI - PI = %.20lf\n", (mypi - PI));
}