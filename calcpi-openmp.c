#include <omp.h>

#include "calcpi.h"

calc_result_t calc_pi(int worksize, int iterations) {
    calc_result_t endResult;
    init_end_result(&endResult);

    starttimer(&endResult.total_time);
    omp_set_num_threads(worksize);
    double mypi = 0.0;
    double n_i;
    double m = 1.0 / (double) iterations;
    //OMP can intelligently figure out whether the variables should be shared or private, but it is good to write
    //it down explicitly anyways
    starttimer(&endResult.calc_time);
#pragma omp parallel for reduction(+: mypi) shared(m) private(n_i)
    for (int i = 0; i < iterations; i++) {
        n_i = ((double) i + 0.5) * m;
        mypi += 4.0 / (1.0 + n_i * n_i);
    }

    mypi *= m;
    endResult.pi_value = mypi;
    stoptimer(&endResult.total_time);
    return endResult;
}
