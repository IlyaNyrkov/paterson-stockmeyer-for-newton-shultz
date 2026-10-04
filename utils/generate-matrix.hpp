#include <vector>
#include <cmath>
#include <random>

template<typename T>
void generate_normalized_matrix(std::vector<T>& mat, int rows, int cols, T phi = 0.5, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<T> unif(0.0, 1.0);
    std::normal_distribution<T> norm(0.0, 1.0);

    double frob_sq = 0.0;
    for (int i = 0; i < rows * cols; ++i) {
        T val = (unif(gen) - 0.5f) * std::exp(phi * norm(gen));
        mat[i] = val;
        frob_sq += (double)val * val;
    }

    // Scale X so ||X||_2 <= ||X||_F = 1.0
    float inv_norm = 1.0f / (float)std::sqrt(frob_sq);
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] *= inv_norm;
    }
}

float calculate_max_relative_error(const std::vector<float>& A, const std::vector<float>& B) {
    float max_err = 0.0f;
    for (size_t i = 0; i < A.size(); ++i) {
        float diff = std::abs(A[i] - B[i]);
        float denom = std::max(std::abs(A[i]), 1e-6f);
        max_err = std::max(max_err, diff / denom);
    }
    return max_err;
}

