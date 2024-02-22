#include "fwd.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_print()
{
    printf("Hello World from GPU1!\n");
    printf("Hello World from GPU2!\n");
    printf("Hello World from GPU3!\n");
}

int hello_world(int argc, char** argv)
{
    test_print<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
