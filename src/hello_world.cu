#include "./fwd.cuh"
#include "./common.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <stdio.h>

__global__ void test_print()
{
    printf("Hello World from GPU1!\n");
    printf("Hello World from GPU2!\n");
    printf("Hello World from GPU3!\n");
}

__global__ void sum_vec(int* a, int* b, int* c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

int hello_world(int argc, char** argv)
{
    int N = 1024;

    std::vector<int> a(N, 11);
    std::vector<int> b(N, 22);
    std::vector<int> c(N, 0);

    int* d_a;
    int* d_b;
    int* d_c;
    CHECK(cudaMalloc(&d_a, N * sizeof(int)));
    CHECK(cudaMalloc(&d_b, N * sizeof(int)));
    CHECK(cudaMalloc(&d_c, N * sizeof(int)));

    CHECK(cudaMemcpy(d_a, a.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b.data(), N * sizeof(int), cudaMemcpyHostToDevice));

    sum_vec<<<N / 32, 32>>>(d_a, d_b, d_c, N);
    CHECK(cudaPeekAtLastError());

    CHECK(cudaMemcpy(c.data(), d_c, N * sizeof(int), cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; ++i) {
        printf("%d + %d = %d\n", a.at(i), b.at(i), c.at(i));
    }

    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    return 0;
}
