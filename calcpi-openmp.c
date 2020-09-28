#include <omp.h>

double calc_pi(int worksize, int iterations) {
    omp_set_num_threads(worksize);
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
    return mypi;
}
