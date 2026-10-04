#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void build_ps_block_kernel(float* B, const float* M, const float* M_powers,
                                      const float* coeffs, int s_actual, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * n;

    if (idx < total_elements) {
        int row = idx % n;
        int col = idx / n;

        float val = (row == col) ? coeffs[0] : 0.0f;

        if (s_actual > 1) {
            val += coeffs[1] * M[idx];
        }

        for (int k = 2; k < s_actual; ++k) {
            int power_offset = (k - 2) * total_elements;
            val += coeffs[k] * M_powers[power_offset + idx];
        }
        B[idx] = val;
    }
}

__global__ void add_scalar_identity_kernel(float* matrix, float c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n * n) {
        int row = idx % n;
        int col = idx / n;
        if (row == col) {
            matrix[idx] += c;
        }
    }
}

struct psWorkspace {
    uint n;
    uint degree;
    uint s;
    uint r;

    float* M;
    float* M_powers;
    float* blocks;
    float* P_M;
    float* temp_gemm;

    psWorkspace(uint polyDegree, uint matrix_N) : degree(polyDegree), n(matrix_N) {
        s = std::ceil(std::sqrt(degree + 1));
        if (s < 2 && degree >= 1) s = 2;
        r = degree / s;

        uint n2 = n * n;

        CHECK_CUDA(cudaMalloc(&M, n2 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&P_M, n2 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&temp_gemm, n2 * sizeof(float)));

        // Correct allocation: s - 1 matrices needed for M^2 through M^s
        if (s >= 2) {
            CHECK_CUDA(cudaMalloc(&M_powers, (s - 1) * n2 * sizeof(float)));
        } else {
            M_powers = nullptr;
        }

        CHECK_CUDA(cudaMalloc(&blocks, (r + 1) * n2 * sizeof(float)));
    }

    ~psWorkspace() {
        cudaFree(M);
        cudaFree(P_M);
        cudaFree(temp_gemm);
        if (M_powers) cudaFree(M_powers);
        cudaFree(blocks);
    }
};

void polynomialPS(cublasHandle_t handle, psWorkspace& ws, const float* d_coeffs) {
    uint n = ws.n;
    uint n2 = n * n;
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Compute powers M^2 through M^s
    if (ws.s >= 2) {
        // M^2 = M * M
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &alpha, ws.M, n, ws.M, n, &beta, ws.M_powers, n));
        // M^k = M * M^{k-1}
        for (uint k = 3; k <= ws.s; ++k) {
            float* prev_power = ws.M_powers + (k - 3) * n2;
            float* next_power = ws.M_powers + (k - 2) * n2;
            CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                     &alpha, ws.M, n, prev_power, n, &beta, next_power, n));
        }
    }

    float* M_s = (ws.s == 1) ? ws.M : (ws.M_powers + (ws.s - 2) * n2);

    int threads = 256;
    int blocks_grid = (n2 + threads - 1) / threads;

    for (uint i = 0; i <= ws.r; ++i) {
        uint s_actual = std::min(ws.s, ws.degree - i * ws.s + 1);
        float* current_block = ws.blocks + i * n2;
        const float* current_coeffs = d_coeffs + i * ws.s;

        build_ps_block_kernel<<<blocks_grid, threads>>>(
                current_block, ws.M, ws.M_powers, current_coeffs, s_actual, n);
    }

    // Horner evaluation on blocks
    CHECK_CUDA(cudaMemcpy(ws.P_M, ws.blocks + ws.r * n2, n2 * sizeof(float), cudaMemcpyDeviceToDevice));

    for (int i = (int)ws.r - 1; i >= 0; --i) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &alpha, ws.P_M, n, M_s, n, &beta, ws.temp_gemm, n));

        CHECK_CUBLAS(cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n,
                                 &alpha, ws.temp_gemm, n, &alpha, ws.blocks + i * n2, n,
                                 ws.P_M, n));
    }
}

void calculateNS_PS(cublasHandle_t handle, psWorkspace& ws, float* d_X, float* d_Res, const float* d_coeffs, uint m) {
    uint n = ws.n;
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 1. M = X^T * X
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &alpha, d_X, m, d_X, m, &beta, ws.M, n));

    // 2. P(M)
    polynomialPS(handle, ws, d_coeffs);

    // 3. Res = X * P(M)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                             &alpha, d_X, m, ws.P_M, n, &beta, d_Res, m));
}

// Workspace for Naive Horner to avoid allocations inside benchmark loops
struct naiveWorkspace {
    float *d_M, *d_PM, *d_temp;
    uint n;
    naiveWorkspace(uint matrix_N) : n(matrix_N) {
        size_t n2 = n * n;
        CHECK_CUDA(cudaMalloc(&d_M, n2 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_PM, n2 * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_temp, n2 * sizeof(float)));
    }
    ~naiveWorkspace() {
        cudaFree(d_M);
        cudaFree(d_PM);
        cudaFree(d_temp);
    }
};

void calculateNS_Naive(cublasHandle_t handle, naiveWorkspace& ws, float* d_X, float* d_Res,
                       const std::vector<float>& h_coeffs, uint m, uint n, uint degree) {
    uint n2 = n * n;
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // 1. M = X^T * X
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m,
                             &alpha, d_X, m, d_X, m, &beta, ws.d_M, n));

    // 2. Horner: P(M) = c_0 I + M(c_1 I + M(...))
    CHECK_CUDA(cudaMemset(ws.d_PM, 0, n2 * sizeof(float)));
    int threads = 256;
    int blocks_grid = (n2 + threads - 1) / threads;
    add_scalar_identity_kernel<<<blocks_grid, threads>>>(ws.d_PM, h_coeffs[degree], n);

    for (int i = (int)degree - 1; i >= 0; --i) {
        CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                                 &alpha, ws.d_PM, n, ws.d_M, n, &beta, ws.d_temp, n));

        CHECK_CUDA(cudaMemcpy(ws.d_PM, ws.d_temp, n2 * sizeof(float), cudaMemcpyDeviceToDevice));
        add_scalar_identity_kernel<<<blocks_grid, threads>>>(ws.d_PM, h_coeffs[i], n);
    }

    // 3. Res = X * P(M)
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n,
                             &alpha, d_X, m, ws.d_PM, n, &beta, d_Res, m));
}