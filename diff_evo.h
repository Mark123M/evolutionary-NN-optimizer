#include <torch/extension.h>

// Declaration of the CUDA function
torch::Tensor square_matrix_cuda(torch::Tensor matrix);

// Wrapper that will be exposed to Python
inline torch::Tensor square_matrix(torch::Tensor matrix) {
    return square_matrix_cuda(matrix);
}