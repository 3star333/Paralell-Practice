/*
 * sgemm.cu
 * Matrix-matrix multiplication C = A*B using CPU and three CUDA kernels.
 *
 * Compile : nvcc -o sgemm sgemm.cu
 * Execute : ./sgemm <m> <k> <n>
 *
 *   A is m x k,  B is k x n,  C is m x n
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

/* ------------------------------------------------------------------ */
/*  Error-checking macro                                               */
/* ------------------------------------------------------------------ */
#define CHECK(call)                                                        \
{                                                                          \
    const cudaError_t error = (call);                                      \
    if (error != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA Error  %s:%d  code=%d  reason=%s\n",        \
                __FILE__, __LINE__, error, cudaGetErrorString(error));     \
        exit(EXIT_FAILURE);                                                \
    }                                                                      \
}

/* ------------------------------------------------------------------ */
/*  CPU wall-clock timer (returns seconds via struct timespec)         */
/*  Returns the current time in seconds as a double.                  */
/* ------------------------------------------------------------------ */
double myCPUTimer(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

/* ------------------------------------------------------------------ */
/*  Host (CPU-only) matrix multiplication                             */
/* ------------------------------------------------------------------ */
void basicSgemm_h(int m, int k, int n,
                  const float *A_h, const float *B_h, float *C_h)
{
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A_h[i * k + l] * B_h[l * n + j];
            }
            C_h[i * n + j] = sum;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  CUDA Kernel 1: each thread computes ONE element of C              */
/* ------------------------------------------------------------------ */
__global__ void matrixMulKernel_1thread1element(int m, int k, int n,
                                                const float *A_d,
                                                const float *B_d,
                                                float *C_d)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A_d[row * k + i] * B_d[i * n + col];
        }
        C_d[row * n + col] = sum;
    }
}

/* ------------------------------------------------------------------ */
/*  CUDA Kernel 2: each thread computes ONE full row of C             */
/* ------------------------------------------------------------------ */
__global__ void matrixMulKernel_1thread1row(int m, int k, int n,
                                            const float *A_d,
                                            const float *B_d,
                                            float *C_d)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                sum += A_d[row * k + i] * B_d[i * n + col];
            }
            C_d[row * n + col] = sum;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  CUDA Kernel 3: each thread computes ONE full column of C          */
/* ------------------------------------------------------------------ */
__global__ void matrixMulKernel_1thread1column(int m, int k, int n,
                                               const float *A_d,
                                               const float *B_d,
                                               float *C_d)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < n) {
        for (int row = 0; row < m; row++) {
            float sum = 0.0f;
            for (int i = 0; i < k; i++) {
                sum += A_d[row * k + i] * B_d[i * n + col];
            }
            C_d[row * n + col] = sum;
        }
    }
}

/* ------------------------------------------------------------------ */
/*  Device wrapper 1: 1 thread  -->  1 element                        */
/*  Returns elapsed kernel time (ms) via CUDA events.                 */
/* ------------------------------------------------------------------ */
void basicSgemm_d_1thread1element(int m, int k, int n,
                                  const float *A_h, const float *B_h,
                                  float *C_h)
{
    float *A_d, *B_d, *C_d;
    size_t sizeA = (size_t)m * k * sizeof(float);
    size_t sizeB = (size_t)k * n * sizeof(float);
    size_t sizeC = (size_t)m * n * sizeof(float);

    CHECK(cudaMalloc((void **)&A_d, sizeA));
    CHECK(cudaMalloc((void **)&B_d, sizeB));
    CHECK(cudaMalloc((void **)&C_d, sizeC));

    CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));

    /* 16x16 thread block; grid covers the full m x n output */
    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x,
              (m + block.y - 1) / block.y);

    cudaEvent_t evStart, evStop;
    CHECK(cudaEventCreate(&evStart));
    CHECK(cudaEventCreate(&evStop));

    CHECK(cudaEventRecord(evStart));
    matrixMulKernel_1thread1element<<<grid, block>>>(m, k, n, A_d, B_d, C_d);
    CHECK(cudaEventRecord(evStop));
    CHECK(cudaEventSynchronize(evStop));

    float kernelMs = 0.0f;
    CHECK(cudaEventElapsedTime(&kernelMs, evStart, evStop));
    printf("  Kernel time (1-thread-1-element) : %.4f ms\n", kernelMs);

    CHECK(cudaEventDestroy(evStart));
    CHECK(cudaEventDestroy(evStop));

    CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/* ------------------------------------------------------------------ */
/*  Device wrapper 2: 1 thread  -->  1 row                            */
/* ------------------------------------------------------------------ */
void basicSgemm_d_1thread1row(int m, int k, int n,
                              const float *A_h, const float *B_h,
                              float *C_h)
{
    float *A_d, *B_d, *C_d;
    size_t sizeA = (size_t)m * k * sizeof(float);
    size_t sizeB = (size_t)k * n * sizeof(float);
    size_t sizeC = (size_t)m * n * sizeof(float);

    CHECK(cudaMalloc((void **)&A_d, sizeA));
    CHECK(cudaMalloc((void **)&B_d, sizeB));
    CHECK(cudaMalloc((void **)&C_d, sizeC));

    CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));

    /* 1-D grid: one thread per row */
    int threadsPerBlock = 256;
    int numBlocks = (m + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t evStart, evStop;
    CHECK(cudaEventCreate(&evStart));
    CHECK(cudaEventCreate(&evStop));

    CHECK(cudaEventRecord(evStart));
    matrixMulKernel_1thread1row<<<numBlocks, threadsPerBlock>>>(m, k, n, A_d, B_d, C_d);
    CHECK(cudaEventRecord(evStop));
    CHECK(cudaEventSynchronize(evStop));

    float kernelMs = 0.0f;
    CHECK(cudaEventElapsedTime(&kernelMs, evStart, evStop));
    printf("  Kernel time (1-thread-1-row)     : %.4f ms\n", kernelMs);

    CHECK(cudaEventDestroy(evStart));
    CHECK(cudaEventDestroy(evStop));

    CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/* ------------------------------------------------------------------ */
/*  Device wrapper 3: 1 thread  -->  1 column                         */
/* ------------------------------------------------------------------ */
void basicSgemm_d_1thread1column(int m, int k, int n,
                                 const float *A_h, const float *B_h,
                                 float *C_h)
{
    float *A_d, *B_d, *C_d;
    size_t sizeA = (size_t)m * k * sizeof(float);
    size_t sizeB = (size_t)k * n * sizeof(float);
    size_t sizeC = (size_t)m * n * sizeof(float);

    CHECK(cudaMalloc((void **)&A_d, sizeA));
    CHECK(cudaMalloc((void **)&B_d, sizeB));
    CHECK(cudaMalloc((void **)&C_d, sizeC));

    CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));

    /* 1-D grid: one thread per column */
    int threadsPerBlock = 256;
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t evStart, evStop;
    CHECK(cudaEventCreate(&evStart));
    CHECK(cudaEventCreate(&evStop));

    CHECK(cudaEventRecord(evStart));
    matrixMulKernel_1thread1column<<<numBlocks, threadsPerBlock>>>(m, k, n, A_d, B_d, C_d);
    CHECK(cudaEventRecord(evStop));
    CHECK(cudaEventSynchronize(evStop));

    float kernelMs = 0.0f;
    CHECK(cudaEventElapsedTime(&kernelMs, evStart, evStop));
    printf("  Kernel time (1-thread-1-column)  : %.4f ms\n", kernelMs);

    CHECK(cudaEventDestroy(evStart));
    CHECK(cudaEventDestroy(evStop));

    CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

/* ------------------------------------------------------------------ */
/*  Verification: compare CPU result with a GPU result element-wise   */
/*  Tolerance chosen to accommodate accumulated float rounding errors */
/* ------------------------------------------------------------------ */
bool verify(float *CPU_Answer, float *GPU_Answer,
            unsigned int nRows, unsigned int nCols)
{
    for (unsigned int i = 0; i < nRows; i++) {
        for (unsigned int j = 0; j < nCols; j++) {
            float diff = fabsf(CPU_Answer[i * nCols + j] -
                               GPU_Answer[i * nCols + j]);
            /* relative tolerance: allow up to 0.1 % error per accumulation */
            float ref  = fabsf(CPU_Answer[i * nCols + j]);
            float tol  = (ref > 1.0f) ? ref * 1e-3f : 1e-3f;
            if (diff > tol) {
                fprintf(stderr,
                        "  Mismatch at [%u][%u]: CPU=%.6f  GPU=%.6f  diff=%.6f\n",
                        i, j, CPU_Answer[i * nCols + j],
                        GPU_Answer[i * nCols + j], diff);
                return false;
            }
        }
    }
    return true;
}

/* ------------------------------------------------------------------ */
/*  main                                                               */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: ./sgemm <m> <k> <n>\n");
        fprintf(stderr, "  A is m x k,  B is k x n,  C is m x n\n");
        return EXIT_FAILURE;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);

    if (m <= 0 || k <= 0 || n <= 0) {
        fprintf(stderr, "All dimensions must be positive integers.\n");
        return EXIT_FAILURE;
    }

    printf("Matrix dimensions:  A(%d x %d)  B(%d x %d)  C(%d x %d)\n\n",
           m, k, k, n, m, n);

    /* ---- Allocate host matrices ---- */
    float *A_h          = (float *)malloc((size_t)m * k * sizeof(float));
    float *B_h          = (float *)malloc((size_t)k * n * sizeof(float));
    float *C_h          = (float *)malloc((size_t)m * n * sizeof(float));
    float *C_GPU_elem   = (float *)malloc((size_t)m * n * sizeof(float));
    float *C_GPU_row    = (float *)malloc((size_t)m * n * sizeof(float));
    float *C_GPU_col    = (float *)malloc((size_t)m * n * sizeof(float));

    if (!A_h || !B_h || !C_h || !C_GPU_elem || !C_GPU_row || !C_GPU_col) {
        fprintf(stderr, "Host malloc failed.\n");
        return EXIT_FAILURE;
    }

    /* ---- Fill A and B with random floats in [0, 1) ---- */
    srand(42u);
    for (int i = 0; i < m * k; i++) A_h[i] = rand() % 100 / 100.0f;
    for (int i = 0; i < k * n; i++) B_h[i] = rand() % 100 / 100.0f;

    /* ---- CPU baseline ---- */
    printf("=== CPU (basicSgemm_h) ===\n");
    double t0 = myCPUTimer();
    basicSgemm_h(m, k, n, A_h, B_h, C_h);
    double t1 = myCPUTimer();
    printf("  Elapsed time                     : %.4f ms\n\n",
           (t1 - t0) * 1000.0);

    /* ---- GPU kernel 1: 1 thread  1 element ---- */
    printf("=== GPU kernel: 1 thread 1 element ===\n");
    t0 = myCPUTimer();
    basicSgemm_d_1thread1element(m, k, n, A_h, B_h, C_GPU_elem);
    t1 = myCPUTimer();
    printf("  Total time (incl. transfers)     : %.4f ms\n", (t1 - t0) * 1000.0);
    if (verify(C_h, C_GPU_elem, (unsigned int)m, (unsigned int)n))
        printf("  Verification vs CPU              : PASSED\n\n");
    else
        printf("  Verification vs CPU              : FAILED\n\n");

    /* ---- GPU kernel 2: 1 thread  1 row ---- */
    printf("=== GPU kernel: 1 thread 1 row ===\n");
    t0 = myCPUTimer();
    basicSgemm_d_1thread1row(m, k, n, A_h, B_h, C_GPU_row);
    t1 = myCPUTimer();
    printf("  Total time (incl. transfers)     : %.4f ms\n", (t1 - t0) * 1000.0);
    if (verify(C_h, C_GPU_row, (unsigned int)m, (unsigned int)n))
        printf("  Verification vs CPU              : PASSED\n\n");
    else
        printf("  Verification vs CPU              : FAILED\n\n");

    /* ---- GPU kernel 3: 1 thread  1 column ---- */
    printf("=== GPU kernel: 1 thread 1 column ===\n");
    t0 = myCPUTimer();
    basicSgemm_d_1thread1column(m, k, n, A_h, B_h, C_GPU_col);
    t1 = myCPUTimer();
    printf("  Total time (incl. transfers)     : %.4f ms\n", (t1 - t0) * 1000.0);
    if (verify(C_h, C_GPU_col, (unsigned int)m, (unsigned int)n))
        printf("  Verification vs CPU              : PASSED\n\n");
    else
        printf("  Verification vs CPU              : FAILED\n\n");

    /* ---- Clean up ---- */
    free(A_h);
    free(B_h);
    free(C_h);
    free(C_GPU_elem);
    free(C_GPU_row);
    free(C_GPU_col);

    return EXIT_SUCCESS;
}