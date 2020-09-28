cmake_minimum_required(VERSION 3.17)
project(2dv605_openmp C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

add_executable(2dv605_openmp main.c timer.h timer.c)