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
const int default_blocks_per_grid = 140;

void print_help(char *cmd) {
#ifdef COMPILE_OPENMP
    printf("Program for calculating PI in parallel, using OpenMP.\n");
#elif COMPILE_CUDA
    printf("Program for calculating PI in parallel, using CUDA.\n");
#endif
    printf("Command format: %s [options]\n", cmd);
    printf("Available options:\n");
    printf("-h        Show this help menu\n");
    printf("-v        Enable verbose printing\n");
    printf("-i [num]  Set the amount of iterations to run. Default: %d\n", default_iterations);
#ifdef COMPILE_OPENMP
    printf("-t [num]  Set the amount of threads to use. Use 0 for maximum available.\n"
           "          Default: amount of threads available on the system (Here: %d)\n", omp_get_max_threads());
#endif
#ifdef COMPILE_CUDA
    printf("-b [num]  Set the batch size for this program. Default: %d\n", default_blocks_per_grid);
#endif
}

int main(int argc, char *argv[]) {
#ifdef COMPILE_OPENMP
    int threads = omp_get_max_threads();
#endif
#ifdef COMPILE_CUDA
    int batch_size = default_blocks_per_grid;
#endif
    int iterations = default_iterations;
    int quiet = 1;
    int option;

#ifdef COMPILE_OPENMP
    const char *opts = ":i:t:vh";
#endif
#ifdef COMPILE_CUDA
    const char *opts = ":i:b:vh";
#endif
    while ((option = getopt(argc, argv, opts)) != -1) {
        switch (option) {
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
#ifdef COMPILE_CUDA
                case 'b': {
                    char end_char;
                    char *end_ptr = &end_char;
                    batch_size = (int) strtol(optarg, &end_ptr, 10);
                    if (iterations < 0 || end_ptr != optarg + strlen(optarg)) {
                        printf("Invalid batch size\n");
                        return 1;
                    }
                    break;
                }
#endif
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
    calc_result_t result;
#ifdef COMPILE_OPENMP
    result = calc_pi(threads, iterations);
#endif
#ifdef COMPILE_CUDA
    result = calc_pi(batch_size, iterations);
#endif
    if (!quiet) {
#ifdef COMPILE_OPENMP
        printf("Performed %d iterations using %d threads.\n", iterations, threads);
#endif
#ifdef COMPILE_CUDA
        printf("Performed %d iterations using %d worksize.\n", iterations, batch_size);
        printf("Memory allocation took %li.%06li seconds\n", result.alloc_time.seconds, result.alloc_time.nanos / 1000);
#endif
        printf("Calculation took %li.%06li seconds\n", result.calc_time.seconds, result.calc_time.nanos / 1000);
        printf("Total time: %li.%06li seconds\n", result.total_time.seconds, result.total_time.nanos / 1000);
        printf("     MyPI = %.20lf\n", result.pi_value);
        printf("MyPI - PI = %.20lf\n", (result.pi_value - PI));
    } else {
        // Output format:
        // With openmp: ts.tmicros, calcs.calcmicros, iter, thrd, pi, diff
        // With cuda: ts.tmicros, calcs.calcmicros, atimes.atimemicros, iter, bsize, pi, diff
#ifdef COMPILE_OPENMP
        printf("%li.%06li, %li.%06li, %d, %d, %.20lf, %.20lf\n", result.total_time.seconds,
               result.total_time.nanos / 1000, result.calc_time.seconds, result.calc_time.nanos / 1000, iterations,
               threads, result.pi_value,
               result.pi_value - PI);
#endif
#ifdef COMPILE_CUDA
        printf("%li.%06li, %li.%06li, %li.%06li, %d, %d, %.20lf, %.20lf\n", result.total_time.seconds,
               result.total_time.nanos / 1000, result.calc_time.seconds, result.calc_time.nanos / 1000,
               result.alloc_time.seconds, result.alloc_time.nanos / 1000, iterations, batch_size, result.pi_value,
               result.pi_value - PI);
#endif
    }
}