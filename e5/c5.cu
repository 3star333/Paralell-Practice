// =============================================================================
//  c5.cu  –  cuBLAS SGEMM with row-major matrices
//
//  cuBLAS is column-major only.  Two approaches are shown to compute C = A·B
//  when A and B are stored in row-major order.
//
//  ── Approach 1 ──────────────────────────────────────────────────────────────
//  Key insight:  a row-major matrix X stored in memory is identical to the
//  column-major representation of Xᵀ.  Therefore:
//
//    row-major A(m×k)  ≡  col-major Aᵀ(k×m)
//    row-major B(k×n)  ≡  col-major Bᵀ(n×k)
//
//  We want C = A·B  ⟹  Cᵀ = Bᵀ · Aᵀ
//
//  Call cublasGemmEx(CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, …, d_B, n, d_A, k, …, d_C, n)
//  cuBLAS computes Cᵀ(n×m) in column-major.
//  Column-major Cᵀ(n×m)  ≡  row-major C(m×n)  →  d_C holds the answer. ✓
//
//  ── Approach 2 ──────────────────────────────────────────────────────────────
//  Explicitly use CUBLAS_OP_T to "undo" the implicit transpose, then transpose
//  the column-major result back to row-major with cublasDgeam.
//
//    d_A as col-major Aᵀ(k×m) + CUBLAS_OP_T  →  A(m×k)
//    d_B as col-major Bᵀ(n×k) + CUBLAS_OP_T  →  B(k×n)
//
//  cublasGemmEx(CUBLAS_OP_T, CUBLAS_OP_T, m, n, k, …, d_A, k, d_B, n, …, d_C_col, m)
//  produces C(m×n) in *column-major* (d_C_col).
//
//  cublasDgeam is then used to transpose d_C_col (col-major m×n) into
//  d_C (col-major n×m ≡ row-major m×n).
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "cublas_utils.h"

using data_type = double;

int main(int argc, char *argv[]) {
    cublasHandle_t cublasH = NULL;
    cudaStream_t stream = NULL;

    // A is m×k,  B is k×n,  C is m×n  (all row-major)
    const int m = 3;
    const int n = 4;
    const int k = 2;

    /*
     *   A (row-major, 3×2)           B (row-major, 2×4)
     *   | 1  2 |                     |  7   8   9  10 |
     *   | 3  4 |                     | 11  12  13  14 |
     *   | 5  6 |
     *
     *   Expected C = A·B (row-major, 3×4)
     *   |  29   32   35   38 |
     *   |  65   72   79   86 |
     *   | 101  112  123  134 |
     */

    const std::vector<data_type> A = {1.0, 2.0,
                                      3.0, 4.0,
                                      5.0, 6.0};

    const std::vector<data_type> B = { 7.0,  8.0,  9.0, 10.0,
                                      11.0, 12.0, 13.0, 14.0};

    std::vector<data_type> C(m * n, 0.0);

    const data_type alpha = 1.0;
    const data_type beta  = 0.0;

    // ── device pointers ───────────────────────────────────────────────────────
    data_type *d_A = nullptr, *d_B = nullptr;
    data_type *d_C = nullptr;          // row-major result (both approaches)
    data_type *d_C_col = nullptr;      // col-major intermediate (Approach 2)

    // ── setup ─────────────────────────────────────────────────────────────────
    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A),
                          m * k * sizeof(data_type)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_B),
                          k * n * sizeof(data_type)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C),
                          m * n * sizeof(data_type)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_C_col),
                          m * n * sizeof(data_type)));

    // Copy row-major A and B to device
    CUBLAS_CHECK(cublasSetVector(m * k, sizeof(data_type),
                                 A.data(), 1, d_A, 1));
    CUBLAS_CHECK(cublasSetVector(k * n, sizeof(data_type),
                                 B.data(), 1, d_B, 1));

    // =========================================================================
    //  APPROACH 1 — swap-and-no-op-transpose trick
    //
    //  cuBLAS call:  Cᵀ(n×m) = Bᵀ(n×k) · Aᵀ(k×m)   [all col-major]
    //  API:  cublasGemmEx(M=n, N=m, K=k,
    //                     "A"=d_B lda=n,  "B"=d_A ldb=k,
    //                     "C"=d_C ldc=n)
    //
    //  d_B is row-major B(k×n) → col-major Bᵀ(n×k), leading dim = n  ✓
    //  d_A is row-major A(m×k) → col-major Aᵀ(k×m), leading dim = k  ✓
    //  d_C will contain col-major Cᵀ(n×m) ≡ row-major C(m×n)         ✓
    // =========================================================================
    printf("=============================================================\n");
    printf(" Approach 1 : swap-and-no-op-transpose  (C = A·B row-major)\n");
    printf("=============================================================\n");

    CUBLAS_CHECK(cublasGemmEx(
        cublasH,
        CUBLAS_OP_N, CUBLAS_OP_N,  // no transpose on either argument
        n, m, k,                   // M=n, N=m, K=k
        &alpha,
        d_B, CUDA_R_64F, n,        // "A" = Bᵀ(n×k) col-major, lda = n
        d_A, CUDA_R_64F, k,        // "B" = Aᵀ(k×m) col-major, ldb = k
        &beta,
        d_C, CUDA_R_64F, n,        // "C" = Cᵀ(n×m) col-major ≡ C(m×n) row-major
        CUDA_R_64F,
        CUBLAS_GEMM_DEFAULT));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // copy result back
    CUBLAS_CHECK(cublasGetVector(m * n, sizeof(data_type),
                                 d_C, 1, C.data(), 1));

    printf("C = A·B\n");
    print_matrix(m, n, C.data(), n);
    printf("\n");

    // =========================================================================
    //  APPROACH 2 — explicit CUBLAS_OP_T + transpose result with cublasDgeam
    //
    //  Step 1: cublasGemmEx with CUBLAS_OP_T on both operands
    //    d_A as col-major Aᵀ(k×m) + OP_T  →  A(m×k),  lda = k
    //    d_B as col-major Bᵀ(n×k) + OP_T  →  B(k×n),  ldb = n
    //    Result d_C_col: col-major C(m×n),  ldc = m
    //
    //  Step 2: cublasDgeam transposes d_C_col (col-major m×n)
    //    → d_C (col-major n×m) ≡ row-major C(m×n)
    // =========================================================================
    printf("=============================================================\n");
    printf(" Approach 2 : CUBLAS_OP_T + cublasDgeam to get row-major C\n");
    printf("=============================================================\n");

    // Step 1 — compute C in column-major
    CUBLAS_CHECK(cublasGemmEx(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_T,  // transpose both: un-does the implicit Aᵀ/Bᵀ
        m, n, k,                   // M=m, N=n, K=k
        &alpha,
        d_A, CUDA_R_64F, k,        // Aᵀ(k×m) col-major, lda=k; OP_T → A(m×k)
        d_B, CUDA_R_64F, n,        // Bᵀ(n×k) col-major, ldb=n; OP_T → B(k×n)
        &beta,
        d_C_col, CUDA_R_64F, m,    // col-major C(m×n), ldc=m
        CUDA_R_64F,
        CUBLAS_GEMM_DEFAULT));

    // Step 2 — transpose col-major C(m×n) → col-major Cᵀ(n×m) ≡ row-major C(m×n)
    //
    //   cublasDgeam: Y(rows×cols) = α·op(A) + β·op(B)   [all col-major]
    //   We want:  d_C(n×m col-major) = 1·Cᵀ(n×m) + 0   where d_C_col is C(m×n)
    //   → OP_T on d_C_col with rows=n, cols=m, lda=m(src), ldc=n(dst)
    const data_type one  = 1.0;
    const data_type zero = 0.0;

    CUBLAS_CHECK(cublasDgeam(
        cublasH,
        CUBLAS_OP_T, CUBLAS_OP_N,  // transpose the source
        n, m,                       // output shape: n rows × m cols (col-major)
        &one,
        d_C_col, m,                 // src C(m×n) col-major, lda = m
        &zero,
        d_C_col, m,                 // unused second operand (β=0)
        d_C,    n));                // dst Cᵀ(n×m) col-major ≡ C(m×n) row-major, ldc = n

    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUBLAS_CHECK(cublasGetVector(m * n, sizeof(data_type),
                                 d_C, 1, C.data(), 1));

    printf("C = A·B\n");
    print_matrix(m, n, C.data(), n);
    printf("\n");

    // ── expected reference ────────────────────────────────────────────────────
    printf("=============================================================\n");
    printf(" Expected C\n");
    printf("=============================================================\n");
    printf("C = A·B\n");
    const std::vector<data_type> C_ref = {
         29.0,  32.0,  35.0,  38.0,
         65.0,  72.0,  79.0,  86.0,
        101.0, 112.0, 123.0, 134.0};
    print_matrix(m, n, C_ref.data(), n);

    // Verify both results match reference
    bool ok = true;
    for (int i = 0; i < m * n; i++)
        if (std::fabs(C[i] - C_ref[i]) > 1e-9) { ok = false; break; }
    printf("\nVerification: %s\n\n", ok ? "PASSED ✓" : "FAILED ✗");

    // ── cleanup ───────────────────────────────────────────────────────────────
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_col));

    CUBLAS_CHECK(cublasDestroy(cublasH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
