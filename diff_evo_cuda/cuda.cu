#include <torch/types.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void de_crossover_kernel(int NP, int num_layers, float CR, float F, int best_model, float** d_layer_ptrs, float** d_bias_ptrs, float** d_out_layer_ptrs, float** d_out_bias_ptrs) {
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
