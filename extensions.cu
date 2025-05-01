#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_cuda {

__global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
	TORCH_CHECK(a.sizes() == b.sizes());
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_CHECK(b.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
	at::Tensor a_contig = a.contiguous();
	at::Tensor b_contig = b.contiguous();
	at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
	const float* a_ptr = a_contig.data_ptr<float>();
	const float* b_ptr = b_contig.data_ptr<float>();
	float* result_ptr = result.data_ptr<float>();

	int numel = a_contig.numel();
	muladd_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, c, result_ptr);
	return result;
}

__global__ void mul_kernel(int numel, const float* a, const float* b, float* result) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numel) result[idx] = a[idx] * b[idx];
}

at::Tensor mymul_cuda(const at::Tensor& a, const at::Tensor& b) {
	TORCH_CHECK(a.sizes() == b.sizes());
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_CHECK(b.dtype() == at::kFloat);
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
	at::Tensor a_contig = a.contiguous();
	at::Tensor b_contig = b.contiguous();
	at::Tensor result = torch::empty(a_contig.sizes(), a_contig.options());
	const float* a_ptr = a_contig.data_ptr<float>();
	const float* b_ptr = b_contig.data_ptr<float>();
	float* result_ptr = result.data_ptr<float>();
	int numel = a_contig.numel();
	mul_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
	return result;
}

__global__ void add_kernel(int numel, const float* a, const float* b, float* result) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < numel) result[idx] = a[idx] * b[idx];
}

void myadd_out_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
	TORCH_CHECK(a.sizes() == b.sizes());
	TORCH_CHECK(b.sizes() == out.sizes());
	TORCH_CHECK(a.dtype() == at::kFloat);
	TORCH_CHECK(b.dtype() == at::kFloat);
	TORCH_CHECK(out.dtype() == at::kFloat);
	TORCH_CHECK(out.is_contiguous());
	TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
	TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);
	at::Tensor a_contig = a.contiguous();
	at::Tensor b_contig = b.contiguous();
	const float* a_ptr = a_contig.data_ptr<float>();
	const float* b_ptr = b_contig.data_ptr<float>();
	float* result_ptr = out.data_ptr<float>();
	int numel = a_contig.numel();
	add_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
}

__global__ void de_crossover_kernel(int NP, int num_layers, float CR, float F, int best_model, float** d_layer_ptrs, float** d_bias_ptrs, float** d_out_layer_ptrs, float** d_out_bias_ptrs) {
	return;
}

std::vector<std::vector<at::Tensor>> de_crossover_cuda(const std::vector<at::Tensor>& layers, const std::vector<at::Tensor>& biases, int64_t NP, double CR, double F, int64_t best_model) {
	int num_layers = layers.size();
	std::vector<const float*> layer_ptrs(num_layers), bias_ptrs(num_layers);
	std::vector<at::Tensor> out_layers(num_layers), out_biases(num_layers);
	std::vector<float*> out_layer_ptrs(num_layers), out_bias_ptrs(num_layers);

	for (int i = 0; i < num_layers; i++) {
		at::Tensor layer_contig = layers[i].contiguous();
		at::Tensor bias_contig = biases[i].contiguous();
		layer_ptrs[i] = layer_contig.data_ptr<float>();
		bias_ptrs[i] = bias_contig.data_ptr<float>();

		out_layers[i] = torch::empty(layer_contig.sizes(), layer_contig.options());
		out_biases[i] = torch::empty(bias_contig.sizes(), bias_contig.options());
		out_layer_ptrs[i] = out_layers[i].data_ptr<float>();
		out_bias_ptrs[i] = out_biases[i].data_ptr<float>();
	}

	float** d_layer_ptrs = nullptr;
	float** d_bias_ptrs = nullptr;
	float** d_out_layer_ptrs = nullptr;
	float** d_out_bias_ptrs = nullptr;

	cudaMalloc(&d_layer_ptrs, num_layers * sizeof(float*));
	cudaMalloc(&d_bias_ptrs, num_layers * sizeof(float*));
	cudaMalloc(&d_out_layer_ptrs, num_layers * sizeof(float*));
	cudaMalloc(&d_out_bias_ptrs, num_layers * sizeof(float*));

	cudaMemcpy(d_layer_ptrs, layer_ptrs.data(), num_layers * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias_ptrs, bias_ptrs.data(), num_layers * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out_layer_ptrs, out_layer_ptrs.data(), num_layers * sizeof(float*), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out_bias_ptrs, out_bias_ptrs.data(), num_layers * sizeof(float*), cudaMemcpyHostToDevice);

	de_crossover_kernel<<<NP / 64, 64>>>(NP, num_layers, CR, F, best_model, d_layer_ptrs, d_bias_ptrs, d_out_layer_ptrs, d_out_bias_ptrs);

	cudaFree(d_layer_ptrs);
	cudaFree(d_bias_ptrs);
	cudaFree(d_out_layer_ptrs);
	cudaFree(d_out_bias_ptrs);

	return {out_layers, out_biases};
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cuda, CUDA, m) {
  m.impl("mymuladd", &mymuladd_cuda);
  m.impl("mymul", &mymul_cuda);
  m.impl("myadd_out", &myadd_out_cuda);
  //m.impl("de_crossover", &de_crossover_cuda);
}

}
