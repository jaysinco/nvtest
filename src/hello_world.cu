#include "./fwd.cuh"
#include "./common.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_print()
{
    printf("Hello World from GPU1!\n");
    printf("Hello World from GPU2!\n");
    printf("Hello World from GPU3!\n");
}

__global__ void add(int a, int b, int* c) { *c = a + b; }

int hello_world(int argc, char** argv)
{
    int* d_c;
    CHECK(cudaMalloc(&d_c, sizeof(int)));

    int a = 2;
    int b = 9;
    add<<<1, 1>>>(a, b, d_c);

    int c;
    CHECK(cudaMemcpy(&c, d_c, sizeof(int), cudaMemcpyDeviceToHost));
    printf("%d+%d=%d\n", a, b, c);
    CHECK(cudaFree(d_c));

    cudaDeviceSynchronize();
    return 0;
}
