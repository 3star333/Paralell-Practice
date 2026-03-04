#define CHECK(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(error); \
    } \
}

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

/* ────────────────────────────────────────────────────────────
 * myCPUTimer – POSIX monotonic wall-clock timer.
 * Call myCPUTimer(NULL) to start, myCPUTimer(&tmr) to read ms.
 * ──────────────────────────────────────────────────────────── */
static struct timespec _cpu_t0;

static inline void myCPUTimer(float *elapsedMs) {
    if (elapsedMs == NULL) {
        /* start */
        clock_gettime(CLOCK_MONOTONIC, &_cpu_t0);
    } else {
        /* stop and compute elapsed ms */
        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        *elapsedMs = (float)((t1.tv_sec  - _cpu_t0.tv_sec)  * 1000.0
                           + (t1.tv_nsec - _cpu_t0.tv_nsec) / 1e6);
    }
}

/* ────────────────────────────────────────────────────────────
 * CUDA-event timer helpers (used inside GPU wrappers)
 * ──────────────────────────────────────────────────────────── */
static inline void cudaTimerStart(cudaEvent_t *start, cudaEvent_t *stop) {
    CHECK(cudaEventCreate(start));
    CHECK(cudaEventCreate(stop));
    CHECK(cudaEventRecord(*start, 0));
}

static inline float cudaTimerStop(cudaEvent_t *start, cudaEvent_t *stop) {
    float ms = 0.0f;
    CHECK(cudaEventRecord(*stop, 0));
    CHECK(cudaEventSynchronize(*stop));
    CHECK(cudaEventElapsedTime(&ms, *start, *stop));
    CHECK(cudaEventDestroy(*start));
    CHECK(cudaEventDestroy(*stop));
    return ms;
}

void basicSgemm_h(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C_h[i * n + j] = 0.0f;
            for (int l = 0; l < k; l++) {
                C_h[i * n + j] += A_h[i * k + l] * B_h[l * n + j];
            }
        }
    }
}

__global__ void matrixMulKernel_1thread1element(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m && col < n) {
        float value = 0.0f;
        for (int i = 0; i < k; i++) {
            value += A_d[row * k + i] * B_d[i * n + col];
        }
        C_d[row * n + col] = value;
    }
}

__global__ void matrixMulKernel_1thread1row(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m) {
        for (int col = 0; col < n; col++) {
            float value = 0.0f;
            for (int i = 0; i < k; i++) {
                value += A_d[row * k + i] * B_d[i * n + col];
            }
            C_d[row * n + col] = value;
        }
    }
}

__global__ void matrixMulKernel_1thread1column(int m, int k, int n, const float *A_d, const float *B_d, float* C_d) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        for (int row = 0; row < m; row++) {
            float value = 0.0f;
            for (int i = 0; i < k; i++) {
                value += A_d[row * k + i] * B_d[i * n + col];
            }
            C_d[row * n + col] = value;
        }
    }
}

void basicSgemm_d_1thread1element(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
    float *A_d, *B_d, *C_d;
    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);
    cudaEvent_t s, e;
    float tMalloc, tH2D, tKernel, tD2H;

    /* cudaMalloc */
    cudaTimerStart(&s, &e);
    CHECK(cudaMalloc((void**)&A_d, sizeA));
    CHECK(cudaMalloc((void**)&B_d, sizeB));
    CHECK(cudaMalloc((void**)&C_d, sizeC));
    tMalloc = cudaTimerStop(&s, &e);

    /* cudaMemcpy Host -> Device */
    cudaTimerStart(&s, &e);
    CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));
    tH2D = cudaTimerStop(&s, &e);

    /* kernel */
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (m + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cudaTimerStart(&s, &e);
    matrixMulKernel_1thread1element<<<numBlocks, threadsPerBlock>>>(m, k, n, A_d, B_d, C_d);
    tKernel = cudaTimerStop(&s, &e);

    /* cudaMemcpy Device -> Host */
    cudaTimerStart(&s, &e);
    CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));
    tD2H = cudaTimerStop(&s, &e);

    printf("  cudaMalloc              : %10.4f ms\n", tMalloc);
    printf("  cudaMemcpy (H->D)       : %10.4f ms\n", tH2D);
    printf("  matrixMulKernel         : %10.4f ms\n", tKernel);
    printf("  cudaMemcpy (D->H)       : %10.4f ms\n", tD2H);
    printf("  sgemm on gpu (total)    : %10.4f ms\n", tMalloc + tH2D + tKernel + tD2H);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void basicSgemm_d_1thread1row(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
    float *A_d, *B_d, *C_d;
    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);
    cudaEvent_t s, e;
    float tMalloc, tH2D, tKernel, tD2H;

    /* cudaMalloc */
    cudaTimerStart(&s, &e);
    CHECK(cudaMalloc((void**)&A_d, sizeA));
    CHECK(cudaMalloc((void**)&B_d, sizeB));
    CHECK(cudaMalloc((void**)&C_d, sizeC));
    tMalloc = cudaTimerStop(&s, &e);

    /* cudaMemcpy Host -> Device */
    cudaTimerStart(&s, &e);
    CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));
    tH2D = cudaTimerStop(&s, &e);

    /* kernel */
    int blocks = (m + 255) / 256;
    cudaTimerStart(&s, &e);
    matrixMulKernel_1thread1row<<<blocks, 256>>>(m, k, n, A_d, B_d, C_d);
    tKernel = cudaTimerStop(&s, &e);

    /* cudaMemcpy Device -> Host */
    cudaTimerStart(&s, &e);
    CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));
    tD2H = cudaTimerStop(&s, &e);

    printf("  cudaMalloc              : %10.4f ms\n", tMalloc);
    printf("  cudaMemcpy (H->D)       : %10.4f ms\n", tH2D);
    printf("  matrixMulKernel         : %10.4f ms\n", tKernel);
    printf("  cudaMemcpy (D->H)       : %10.4f ms\n", tD2H);
    printf("  sgemm on gpu (total)    : %10.4f ms\n", tMalloc + tH2D + tKernel + tD2H);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

void basicSgemm_d_1thread1column(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
    float *A_d, *B_d, *C_d;
    size_t sizeA = m * k * sizeof(float);
    size_t sizeB = k * n * sizeof(float);
    size_t sizeC = m * n * sizeof(float);
    cudaEvent_t s, e;
    float tMalloc, tH2D, tKernel, tD2H;

    /* cudaMalloc */
    cudaTimerStart(&s, &e);
    CHECK(cudaMalloc((void**)&A_d, sizeA));
    CHECK(cudaMalloc((void**)&B_d, sizeB));
    CHECK(cudaMalloc((void**)&C_d, sizeC));
    tMalloc = cudaTimerStop(&s, &e);

    /* cudaMemcpy Host -> Device */
    cudaTimerStart(&s, &e);
    CHECK(cudaMemcpy(A_d, A_h, sizeA, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B_d, B_h, sizeB, cudaMemcpyHostToDevice));
    tH2D = cudaTimerStop(&s, &e);

    /* kernel */
    int blocks = (n + 255) / 256;
    cudaTimerStart(&s, &e);
    matrixMulKernel_1thread1column<<<blocks, 256>>>(m, k, n, A_d, B_d, C_d);
    tKernel = cudaTimerStop(&s, &e);

    /* cudaMemcpy Device -> Host */
    cudaTimerStart(&s, &e);
    CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));
    tD2H = cudaTimerStop(&s, &e);

    printf("  cudaMalloc              : %10.4f ms\n", tMalloc);
    printf("  cudaMemcpy (H->D)       : %10.4f ms\n", tH2D);
    printf("  matrixMulKernel         : %10.4f ms\n", tKernel);
    printf("  cudaMemcpy (D->H)       : %10.4f ms\n", tD2H);
    printf("  sgemm on gpu (total)    : %10.4f ms\n", tMalloc + tH2D + tKernel + tD2H);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols) {
    for (unsigned int i = 0; i < nRows; i++) {
        for (unsigned int j = 0; j < nCols; j++) {
            if (fabs(CPU_Answer[i * nCols + j] - GPU_Answer[i * nCols + j]) > 1e-5) {
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: ./sgemm <m> <k> <n>\n");
        return -1;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);

    float *A_h        = (float*)malloc(m * k * sizeof(float));
    float *B_h        = (float*)malloc(k * n * sizeof(float));
    float *C_h        = (float*)malloc(m * n * sizeof(float));
    float *C_GPU_1element = (float*)malloc(m * n * sizeof(float));
    float *C_GPU_1row     = (float*)malloc(m * n * sizeof(float));
    float *C_GPU_1column  = (float*)malloc(m * n * sizeof(float));

    for (int i = 0; i < m * k; i++) A_h[i] = rand() % 100 / 100.0f;
    for (int i = 0; i < k * n; i++) B_h[i] = rand() % 100 / 100.0f;

    float cpuMs;

    printf("============================================================\n");
    printf("Matrix size: %d x %d x %d  (m x k x n)\n", m, k, n);
    printf("============================================================\n");

    /* ── CPU host function ── */
    myCPUTimer(NULL);
    basicSgemm_h(m, k, n, A_h, B_h, C_h);
    myCPUTimer(&cpuMs);
    printf("sgemm on cpu                : %10.4f ms\n", cpuMs);

    /* ── GPU: 1 thread / element ── */
    printf("------------------------------------------------------------\n");
    printf("[1 thread per element]\n");
    basicSgemm_d_1thread1element(m, k, n, A_h, B_h, C_GPU_1element);
    printf("  verifying results....     : %s\n",
           verify(C_h, C_GPU_1element, m, n) ? "PASSED" : "FAILED");

    /* ── GPU: 1 thread / row ── */
    printf("------------------------------------------------------------\n");
    printf("[1 thread per row]\n");
    basicSgemm_d_1thread1row(m, k, n, A_h, B_h, C_GPU_1row);
    printf("  verifying results....     : %s\n",
           verify(C_h, C_GPU_1row, m, n) ? "PASSED" : "FAILED");

    /* ── GPU: 1 thread / column ── */
    printf("------------------------------------------------------------\n");
    printf("[1 thread per column]\n");
    basicSgemm_d_1thread1column(m, k, n, A_h, B_h, C_GPU_1column);
    printf("  verifying results....     : %s\n",
           verify(C_h, C_GPU_1column, m, n) ? "PASSED" : "FAILED");

    printf("============================================================\n");

    free(A_h);
    free(B_h);
    free(C_h);
    free(C_GPU_1element);
    free(C_GPU_1row);
    free(C_GPU_1column);

    return 0;
}