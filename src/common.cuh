#pragma once
#include <sys/time.h>

#define CHECK(call)                                                                      \
    {                                                                                    \
        const cudaError_t error = call;                                                  \
        if (error != cudaSuccess) {                                                      \
            fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                       \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        }                                                                                \
    }

inline double seconds()
{
    timeval tp;
    struct timezone tzp;
    int i = gettimeofday(&tp, &tzp);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
