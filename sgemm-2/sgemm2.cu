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
#include <math.h>
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

/* ─────────────────────────────────────────────────────────────
 * matrixMulKernel_tiled — tiled kernel with dynamic shared memory
 *
 * Shared memory layout (single extern array split into two tiles):
 *   [ As (Adz_sz bytes) | Bs (Bdz_sz bytes) ]
 *
 * Adz_sz = tileSize * tileSize * sizeof(float)
 * Bdz_sz = tileSize * tileSize * sizeof(float)
 * Total passed as 3rd kernel launch parameter <<<grid, block, sharedBytes>>>
 *
 * Boundary conditions: threads outside matrix bounds load 0.0f so
 * partial edge tiles are handled correctly for any m, k, n.
 * ───────────────────────────────────────────────────────────── */
__global__ void matrixMulKernel_tiled(int m, int k, int n,
    const float *A_d, const float *B_d, float *C_d,
    unsigned Adz_sz, unsigned Bdz_sz)
{
    extern __shared__ float sharedMem[];
    float *As = sharedMem;                             /* tile of A */
    float *Bs = sharedMem + (Adz_sz / sizeof(float)); /* tile of B */

    int tileSize = blockDim.x;   /* blockDim.x == blockDim.y */
    int row = blockIdx.y * tileSize + threadIdx.y;
    int col = blockIdx.x * tileSize + threadIdx.x;
    float value = 0.0f;

    int numTiles = (k + tileSize - 1) / tileSize;

    for (int t = 0; t < numTiles; t++) {
        /* load tile of A — coalesced across warp */
        int aCol = t * tileSize + threadIdx.x;
        As[threadIdx.y * tileSize + threadIdx.x] =
            (row < m && aCol < k) ? A_d[row * k + aCol] : 0.0f;

        /* load tile of B — coalesced across warp */
        int bRow = t * tileSize + threadIdx.y;
        Bs[threadIdx.y * tileSize + threadIdx.x] =
            (bRow < k && col < n) ? B_d[bRow * n + col] : 0.0f;

        __syncthreads();

        for (int i = 0; i < tileSize; i++)
            value += As[threadIdx.y * tileSize + i] * Bs[i * tileSize + threadIdx.x];

        __syncthreads();
    }

    if (row < m && col < n)
        C_d[row * n + col] = value;
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
    CHECK(cudaGetLastError());
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

void basicSgemm_d_tiled(int m, int k, int n, const float *A_h, const float *B_h, float* C_h) {
    /* ── device query: max shared memory per block ── */
    int devId;
    CHECK(cudaGetDevice(&devId));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, devId));
    size_t maxShared = prop.sharedMemPerBlock;  /* e.g. 49152 bytes on V100/A100 */

    /*
     * Largest tileSize s.t. two tiles (As + Bs) fit in shared memory:
     *   2 * tileSize^2 * sizeof(float) <= maxShared
     * Round down to nearest multiple of 16 for warp alignment.
     */
    int tileSize = (int)sqrtf((float)(maxShared / (2 * sizeof(float))));
    tileSize = (tileSize / 16) * 16;
    if (tileSize < 16) tileSize = 16;  /* safety floor */

    unsigned Adz_sz    = (unsigned)(tileSize * tileSize * sizeof(float));
    unsigned Bdz_sz    = (unsigned)(tileSize * tileSize * sizeof(float));
    size_t sharedBytes = Adz_sz + Bdz_sz;  /* 3rd kernel launch parameter */

    printf("  [device: %s | maxShared: %zu B | tileSize: %dx%d]\n",
           prop.name, maxShared, tileSize, tileSize);

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

    /* kernel launch — sharedBytes passed as 3rd parameter */
    dim3 threadsPerBlock(tileSize, tileSize);
    dim3 numBlocks((n + tileSize - 1) / tileSize,
                   (m + tileSize - 1) / tileSize);
    cudaTimerStart(&s, &e);
    matrixMulKernel_tiled<<<numBlocks, threadsPerBlock, sharedBytes>>>(
        m, k, n, A_d, B_d, C_d, Adz_sz, Bdz_sz);
    CHECK(cudaGetLastError());
    tKernel = cudaTimerStop(&s, &e);

    /* cudaMemcpy Device -> Host */
    cudaTimerStart(&s, &e);
    CHECK(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));
    tD2H = cudaTimerStop(&s, &e);

    printf("  cudaMalloc              : %10.4f ms\n", tMalloc);
    printf("  cudaMemcpy (H->D)       : %10.4f ms\n", tH2D);
    printf("  matrixMulKernel (tiled) : %10.4f ms\n", tKernel);
    printf("  cudaMemcpy (D->H)       : %10.4f ms\n", tD2H);
    printf("  sgemm on gpu (total)    : %10.4f ms\n", tMalloc + tH2D + tKernel + tD2H);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}


bool verify(float* CPU_Answer, float* GPU_Answer, unsigned int nRows, unsigned int nCols) {
    for (unsigned int i = 0; i < nRows; i++) {
        for (unsigned int j = 0; j < nCols; j++) {
            float ref  = fabs(CPU_Answer[i * nCols + j]);
            float diff = fabs(CPU_Answer[i * nCols + j] - GPU_Answer[i * nCols + j]);
            /* relative tolerance of 0.1% for large values, absolute 1e-3 for near-zero */
            float tol  = (ref > 1.0f) ? ref * 1e-3f : 1e-3f;
            if (diff > tol) {
                printf("  MISMATCH at [%u][%u]: cpu=%.6f  gpu=%.6f  diff=%.6f\n",
                       i, j, CPU_Answer[i * nCols + j], GPU_Answer[i * nCols + j], diff);
                return false;
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: ./sgemm2 <m> <k> <n>\n");
        return -1;
    }

    int m = atoi(argv[1]);
    int k = atoi(argv[2]);
    int n = atoi(argv[3]);

    srand((unsigned int)time(NULL));

    float *A_h           = (float*)malloc(m * k * sizeof(float));
    float *B_h           = (float*)malloc(k * n * sizeof(float));
    float *C_h           = (float*)malloc(m * n * sizeof(float));
    float *C_GPU_element = (float*)malloc(m * n * sizeof(float));
    float *C_GPU_tiled   = (float*)malloc(m * n * sizeof(float));

    if (!A_h || !B_h || !C_h || !C_GPU_element || !C_GPU_tiled) {
        fprintf(stderr, "Error: host malloc failed\n");
        return -1;
    }

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

    /* ── GPU: 1 thread per element ── */
    printf("------------------------------------------------------------\n");
    printf("[1 thread per element]\n");
    basicSgemm_d_1thread1element(m, k, n, A_h, B_h, C_GPU_element);
    printf("  verifying results....     : %s\n",
           verify(C_h, C_GPU_element, m, n) ? "PASSED" : "FAILED");

    /* ── GPU: tiled shared memory ── */
    printf("------------------------------------------------------------\n");
    printf("[tiled shared memory]\n");
    basicSgemm_d_tiled(m, k, n, A_h, B_h, C_GPU_tiled);
    printf("  verifying results....     : %s\n",
           verify(C_h, C_GPU_tiled, m, n) ? "PASSED" : "FAILED");

    printf("============================================================\n");

    free(A_h);
    free(B_h);
    free(C_h);
    free(C_GPU_element);
    free(C_GPU_tiled);

    return 0;
}