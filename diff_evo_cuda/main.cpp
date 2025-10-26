#include <torch/extension.h>

void pcg_init(int max_size);
void pcg_destroy();
void curand_init();
std::vector<std::vector<torch::Tensor>> de_crossover_cuda(const std::vector<torch::Tensor>& layers, const std::vector<torch::Tensor>& biases, int64_t NP, double CR, double F, int64_t best_model);
std::vector<std::vector<torch::Tensor>> de_crossover_cuda2(const std::vector<torch::Tensor>& layers, const std::vector<torch::Tensor>& biases, int64_t NP, double CR, double F, const torch::Tensor& best_model);
void de_update_cuda(int NP, std::vector<torch::Tensor>& layers, std::vector<torch::Tensor>& biases, const std::vector<torch::Tensor>& y_layers, const std::vector<torch::Tensor>& y_biases, torch::Tensor& fx, const torch::Tensor& fy, torch::Tensor& min_f, torch::Tensor& best_model);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
m.def("de_crossover_cuda", torch::wrap_pybind_function(de_crossover_cuda), "de_crossover_cuda");
m.def("de_crossover_cuda2", torch::wrap_pybind_function(de_crossover_cuda2), "de_crossover_cuda2");
m.def("de_update_cuda", torch::wrap_pybind_function(de_update_cuda), "de_update_cuda");
m.def("curand_init", torch::wrap_pybind_function(curand_init), "curand_init");
m.def("pcg_init", torch::wrap_pybind_function(pcg_init), "pcg_init");
m.def("pcg_destroy", torch::wrap_pybind_function(pcg_destroy), "pcg_destroy");
}