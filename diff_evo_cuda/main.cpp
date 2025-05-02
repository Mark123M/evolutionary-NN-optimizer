#include <torch/extension.h>

void pcg32_fast_init(uint64_t seed);
std::vector<std::vector<torch::Tensor>> de_crossover_cuda(const std::vector<torch::Tensor>& layers, const std::vector<torch::Tensor>& biases, int64_t NP, double CR, double F, int64_t best_model);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("de_crossover_cuda", torch::wrap_pybind_function(de_crossover_cuda), "de_crossover_cuda");
m.def("pcg32_fast_init", torch::wrap_pybind_function(pcg32_fast_init), "pcg32_fast_init");
}