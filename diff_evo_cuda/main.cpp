#include <torch/extension.h>

void curand_init();
std::vector<std::vector<torch::Tensor>> de_crossover_cuda(const std::vector<torch::Tensor>& layers, const std::vector<torch::Tensor>& biases, int64_t NP, double CR, double F, int64_t best_model);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("de_crossover_cuda", torch::wrap_pybind_function(de_crossover_cuda), "de_crossover_cuda");
m.def("curand_init", torch::wrap_pybind_function(curand_init), "curand_init");
}