cmake_minimum_required(VERSION 3.17)
project(2dv605_openmp C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")

add_executable(2dv605_openmp main.c timer.h timer.c calcpi.h calcpi-openmp.c calcpi.c)
target_compile_options(2dv605_openmp PRIVATE $<$<COMPILE_LANGUAGE:C>:-DCOMPILE_OPENMP>)

# Disable the CUDA project if a CUDA compiler cannot be found
if (CMAKE_CUDA_COMPILER)
    project(2dv605_cuda CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_HOST_COMPILER nvcc)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DCOMPILE_CUDA")
    add_executable(2dv605_cuda main.c timer.h timer.c calcpi.h calcpi-cuda.cu calcpi.c)
    target_compile_options(2dv605_cuda PRIVATE $<$<COMPILE_LANGUAGE:C>:-DCOMPILE_CUDA>)
endif ()
