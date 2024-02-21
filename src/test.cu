#include "test.cuh"
#include <stdio.h>

__global__ void test_print()
{
    ;
    printf("Hello World from GPU!\n");
}

void wrap_test_print()
{
    test_print<<<1, 1>>>();
    cudaDeviceReset();
    return;
}