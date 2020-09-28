#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>

#ifdef COMPILE_OPENMP

#include <omp.h>

#endif

#include "calcpi.h"
#include "timer.h"

#define PI 3.14159265358979323846264

// For this task, 4x oversubscription seems to actually improve program speed? Theorize about non-uniform memory access
// causing lots of memory-waiting. Run tests!

const int default_iterations = 192000000;

void print_help(char *cmd) {
    printf("Program for calculating PI in parallel, using OpenMP.\n");
    printf("Command format: %s [options]\n", cmd);
    printf("Available options:\n");
    printf("-h        Show this help menu\n");
    printf("-v        Enable verbose printing\n");
#ifdef COMPILE_OPENMP
    printf("-t [num]  Set the amount of threads to use. Use 0 for maximum available.\n"
           "          Default: amount of threads available on the system (Here: %d)\n", omp_get_max_threads());
#endif
    printf("-i [num]  Set the amount of iterations to run. Default: %d\n", default_iterations);
}

int main(int argc, char *argv[]) {
    int iterations = default_iterations;
#ifdef COMPILE_OPENMP
    int threads = omp_get_max_threads();
#endif
    int quiet = 1;
    int option;
#ifdef COMPILE_OPENMP
    const char* opts = ":i:t:vh";
#else
    const char* opts = ":i:vh";
#endif
    while ((option = getopt(argc, argv, opts)) != -1) {
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
#ifdef COMPILE_OPENMP
            case 't': {
                char end_char;
                char *end_ptr = &end_char;
                threads = (int) strtol(optarg, &end_ptr, 10);
                if (iterations < 0 || end_ptr != optarg + strlen(optarg)) {
                    printf("Invalid thread count\n");
                    return 1;
                }
                if (threads == 0) {
                    threads = omp_get_max_threads();
                }
                break;
            }
#endif
            case 'v':
                quiet = 0;
                break;
            case 'h':
                print_help(argv[0]);
                return 0;
            case '?':
            default: {
                printf("Unknown option %c\n", optopt);
                print_help(argv[0]);
                return 1;
            }
        }
    }
#ifdef COMPILE_OPENMP
    omp_set_num_threads(threads);
#endif

    timekeeper_t timer;
    starttimer(&timer);
    double mypi = calc_pi(iterations);
    stoptimer(&timer);
    if (!quiet) {
        printf("Computation took %li.%06li seconds\n", timer.seconds, timer.nanos / 1000);
#ifdef COMPILE_OPENMP
        printf("Performed %d iterations using %d threads.\n", iterations, threads);
#else
        printf("Performed %d iterations.\n", iterations);
#endif
        printf("     MyPI = %.20lf\n", mypi);
        printf("MyPI - PI = %.20lf\n", (mypi - PI));
    } else {
        // Output format:
        // With openmp: s.micros, iter, thrd, pi, diff
        // With cuda:
#ifdef COMPILE_OPENMP
        printf("%li.%06li, %d, %d, %.20lf, %.20lf\n", timer.seconds, timer.nanos / 1000, iterations, threads, mypi,
               mypi - PI);
#else
        printf("%li.%06li, %d, %.20lf, %.20lf\n", timer.seconds, timer.nanos / 1000, iterations, mypi,
               mypi - PI);
#endif
    }
}