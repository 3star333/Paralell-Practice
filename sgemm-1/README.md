# Matrix-Matrix Multiplication using CUDA

This project implements matrix-matrix multiplication using CUDA C. The program computes the product of two matrices, C = A * B, where A is of size m x k, B is of size k x n, and C is of size m x n.

## Files

- **sgemm.cu**: Contains the complete CUDA C program for matrix-matrix multiplication, including:
  - Error checking macro.
  - CPU-only matrix multiplication function.
  - CUDA kernels for different computation strategies.
  - Host functions for memory management and kernel invocation.
  - Main function to handle execution flow.
  - Verification function to ensure correctness of results.

## Compilation and Execution

To compile the program, use the following command:

```
nvcc -o sgemm sgemm.cu
```

To execute the program, run:

```
./sgemm <m> <k> <n>
```

Where `<m>`, `<k>`, and `<n>` specify the dimensions of matrices A, B, and C respectively.

## Functionality

The program includes the following key components:

1. **Error Checking**: A macro `CHECK(call)` is defined to handle CUDA errors.
2. **Timing**: A function `myCPUTimer()` is implemented to measure execution time for performance evaluation.
3. **Matrix Multiplication**:
   - **CPU Implementation**: The function `basicSgemm_h()` performs matrix multiplication on the CPU.
   - **CUDA Kernels**:
     - `matrixMulKernel_1thread1element()`: Each thread computes one element of the output matrix.
     - `matrixMulKernel_1thread1row()`: Each thread computes one row of the output matrix.
     - `matrixMulKernel_1thread1column()`: Each thread computes one column of the output matrix.
4. **Host Functions**: Functions to manage device memory and invoke the respective CUDA kernels.
5. **Verification**: A function `verify()` checks if the results from the CPU and GPU implementations match.

## Performance Evaluation

The program measures the execution time for the CPU and GPU implementations, allowing for performance comparisons across different matrix sizes and multiplication strategies.

## Notes

- Ensure that your system has a compatible NVIDIA GPU and the CUDA toolkit installed to run this program.
- The program handles varying input dimensions and accounts for boundary conditions in matrix multiplication.