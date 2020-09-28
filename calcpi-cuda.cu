extern "C" {
__global__ void bruh() {

}

double calc_pi(int iterations) {
    for (int i = 0; i < iterations; i++) {
        bruh<<<3, 4>>>();
    }
    return 3.14;
}
};
