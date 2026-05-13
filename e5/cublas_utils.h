#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

// ─── Error-checking macros ────────────────────────────────────────────────────

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t _e = (err);                                                \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));               \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(err)                                                      \
    do {                                                                       \
        cublasStatus_t _s = (err);                                             \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr, "cuBLAS error %s:%d: %d\n",                       \
                    __FILE__, __LINE__, (int)_s);                              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// ─── Print a row-major matrix ─────────────────────────────────────────────────
//   rows × cols, stored with leading dimension 'ld' (number of columns)

static inline void print_matrix(int rows, int cols,
                                 const double *data, int /*ld*/)
{
    for (int r = 0; r < rows; r++) {
        printf("  |");
        for (int c = 0; c < cols; c++)
            printf(" %8.1f", data[r * cols + c]);
        printf(" |\n");
    }
}
