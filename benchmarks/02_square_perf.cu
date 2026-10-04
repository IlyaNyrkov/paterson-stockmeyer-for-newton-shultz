#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../src/paterson-stockmeyer.cuh"
#include "../utils/generate-matrix.hpp"


int main() {
    const std::vector<int> matrix_sizes = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    const int warmup_runs = 3;
    const int benchmark_runs = 10;
    const float EPSILON = 5e-3f;

    // Collection of Taylor coefficients for Newton-Schulz Orders
    std::vector<std::vector<float>> test_configs = {
            {1.0f, -0.5f},                                               // Order-2 NS (Degree 3 in X)
            {1.0f, -0.5f, 0.375f},                                       // Order-3 NS (Degree 5 in X)
            {1.0f, -0.5f, 0.375f, -0.3125f, 0.2734f},                    // Order-5 NS (Degree 9 in X)
            {1.0f, -0.5f, 0.375f, -0.3125f, 0.2734f, -0.2461f, 0.2256f}, // Order-7 NS (Degree 13 in X)
            {1.0f, -0.5f, 0.375f, -0.3125f, 0.2734f, -0.2461f, 0.2256f, -0.2095f, 0.1964f} // Order-9 NS (Degree 17 in X)
    };

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    std::cout << "Starting Benchmark (Square Matrices)" << std::endl;

    for (const auto& h_coeffs : test_configs) {
        uint degree = h_coeffs.size() - 1;
        uint order = degree + 1;
        uint overall_degree_x = 2 * order - 1;

        std::cout << "\n=======================================================" << std::endl;
        std::cout << "Order-" << order << " Newton-Schulz (Overall Degree " << overall_degree_x << " in X)" << std::endl;
        std::cout << "Size,Naive_Time_ms,PS_Time_ms,Max_Rel_Error,Match_Status" << std::endl;

        for (int n : matrix_sizes) {
            size_t matrix_elements = (size_t)n * n;
            size_t matrix_bytes = matrix_elements * sizeof(float);

            std::vector<float> h_X(matrix_elements);
            std::vector<float> h_Res_Naive(matrix_elements);
            std::vector<float> h_Res_PS(matrix_elements);

            generate_normalized_matrix(h_X, n, n, 0.5f, 42);

            float *d_X, *d_Res_Naive, *d_Res_PS, *d_coeffs;
            CHECK_CUDA(cudaMalloc(&d_X, matrix_bytes));
            CHECK_CUDA(cudaMalloc(&d_Res_Naive, matrix_bytes));
            CHECK_CUDA(cudaMalloc(&d_Res_PS, matrix_bytes));
            CHECK_CUDA(cudaMalloc(&d_coeffs, h_coeffs.size() * sizeof(float)));

            CHECK_CUDA(cudaMemcpy(d_X, h_X.data(), matrix_bytes, cudaMemcpyHostToDevice));
            CHECK_CUDA(cudaMemcpy(d_coeffs, h_coeffs.data(), h_coeffs.size() * sizeof(float), cudaMemcpyHostToDevice));

            psWorkspace ws(degree, n);
            naiveWorkspace n_ws(n);

            // 1. Benchmark Naive
            for (int i = 0; i < warmup_runs; ++i) {
                calculateNS_Naive(handle, n_ws, d_X, d_Res_Naive, h_coeffs, n, n, degree);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < benchmark_runs; ++i) {
                calculateNS_Naive(handle, n_ws, d_X, d_Res_Naive, h_coeffs, n, n, degree);
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float naive_ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&naive_ms, start, stop));
            naive_ms /= benchmark_runs;

            // 2. Benchmark Paterson-Stockmeyer
            for (int i = 0; i < warmup_runs; ++i) {
                calculateNS_PS(handle, ws, d_X, d_Res_PS, d_coeffs, n);
            }
            CHECK_CUDA(cudaDeviceSynchronize());

            CHECK_CUDA(cudaEventRecord(start));
            for (int i = 0; i < benchmark_runs; ++i) {
                calculateNS_PS(handle, ws, d_X, d_Res_PS, d_coeffs, n);
            }
            CHECK_CUDA(cudaEventRecord(stop));
            CHECK_CUDA(cudaEventSynchronize(stop));

            float ps_ms = 0;
            CHECK_CUDA(cudaEventElapsedTime(&ps_ms, start, stop));
            ps_ms /= benchmark_runs;

            // 3. Verification
            CHECK_CUDA(cudaMemcpy(h_Res_Naive.data(), d_Res_Naive, matrix_bytes, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_Res_PS.data(), d_Res_PS, matrix_bytes, cudaMemcpyDeviceToHost));

            float rel_error = calculate_max_relative_error(h_Res_Naive, h_Res_PS);
            std::string status = (rel_error <= EPSILON) ? "PASS" : "FAIL";

            std::cout << std::fixed << std::setprecision(4)
                      << n << ","
                      << naive_ms << ","
                      << ps_ms << ","
                      << std::scientific << std::setprecision(2) << rel_error << ","
                      << status << std::endl;

            CHECK_CUDA(cudaFree(d_X));
            CHECK_CUDA(cudaFree(d_Res_Naive));
            CHECK_CUDA(cudaFree(d_Res_PS));
            CHECK_CUDA(cudaFree(d_coeffs));
        }
    }

    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}