#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ uint64_t mcg_state = 0xcafef00dd15ea5e5u;	// Must be odd

__device__ uint32_t pcg32_fast() {
	uint64_t x = mcg_state;
	unsigned count = (unsigned)(x >> 61);	// 61 = 64 - 3

	mcg_state = x * 6364136223846793005u;
	x ^= x >> 22;
	return (uint32_t)(x >> (22 + count));	// 22 = 32 - 3 - 7
}

void pcg32_fast_init(uint64_t seed) {
	uint64_t host_state = 2 * seed + 1;
    cudaMemcpyToSymbol(mcg_state, &host_state, sizeof(uint64_t));
    std::cout << host_state << std::endl;
}

__global__ void de_crossover_kernel(int NP, int num_layers, float CR, float F, int best_model, float** d_layer_ptrs, float** d_bias_ptrs, float** d_out_layer_ptrs, float** d_out_bias_ptrs) {
	int id = blockIdx.x * blockDim.x + threadIdx.x; // candidate id
	for (int h = 0; h < 3; h++) {
		uint32_t agent_id = pcg32_fast();
	}
	return;
}

std::vector<std::vector<torch::Tensor>> de_crossover_cuda(const std::vector<torch::Tensor>& layers, const std::vector<torch::Tensor>& biases, int64_t NP, double CR, double F, int64_t best_model) {
	int num_layers = layers.size();
	std::vector<const float*> layer_ptrs(num_layers), bias_ptrs(num_layers);
	std::vector<torch::Tensor> out_layers(num_layers), out_biases(num_layers);
	std::vector<float*> out_layer_ptrs(num_layers), out_bias_ptrs(num_layers);

	for (int i = 0; i < num_layers; i++) {
		torch::Tensor layer_contig = layers[i].contiguous();
		torch::Tensor bias_contig = biases[i].contiguous();
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

	de_crossover_kernel<<<max(1l, NP / 64), 64>>>(NP, num_layers, CR, F, best_model, d_layer_ptrs, d_bias_ptrs, d_out_layer_ptrs, d_out_bias_ptrs);

	cudaFree(d_layer_ptrs);
	cudaFree(d_bias_ptrs);
	cudaFree(d_out_layer_ptrs);
	cudaFree(d_out_bias_ptrs);

	return {out_layers, out_biases};
}
