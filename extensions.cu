#include <vector>
#include <torch/extension.h>
#include <curand>

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
}

/* fastest unbiased
uint32_t bounded_rand(uint32_t range) {
    uint32_t x = pcg32_fast();
    uint64_t m = uint64_t(x) * uint64_t(range);
    uint32_t l = uint32_t(m);
    if (l < range) {
        uint32_t t = -range;
        if (t >= range) {
            t -= range;
            if (t >= range) 
                t %= range;
        }
        while (l < t) {
            x = pcg32_fast();
            m = uint64_t(x) * uint64_t(range);
            l = uint32_t(m);
        }
    }
    return m >> 32;
} */

__device__ uint32_t bounded_rand(uint32_t range) {
    return pcg32_fast() % range;
}


__global__ void de_crossover_step(int id, uint32_t best_model, float F, float* d_layer, float* d_bias, float* d_out_layer, float* d_out_bias) {

}

__global__ void de_crossover_kernel(uint32_t NP, uint32_t num_layers, uint32_t CR, float F, uint32_t best_model, float** d_layer_ptrs, float** d_bias_ptrs, float** d_out_layer_ptrs, float** d_out_bias_ptrs) {
	int id = blockIdx.x * blockDim.x + threadIdx.x; // candidate id
	uint32_t agent_ids[3]{bounded_rand(NP), bounded_rand(NP), bounded_rand(NP)};
	uint32_t R = bounded_rand(num_layers);
	for (int i = 0; i < num_layers; i++) {
		uint32_t ri = pcg32_fast();
		if (ri < CR || i == R) {
			// Perform crossover
			de_crossover_step<<<max(1, NP / 64), 64>>>(id, best_model, F, d_layer_ptrs[id], d_bias_ptrs[id], d_out_layer_ptrs[id], d_out_bias_ptrs[id]);
		}
	}
	return;
}

std::vector<std::vector<torch::Tensor>> de_crossover_cuda(const std::vector<torch::Tensor>& layers, const std::vector<torch::Tensor>& biases, int64_t NP, double CR, double F, int64_t best_model) {
	uint32_t num_layers = layers.size();
	std::vector<const float*> layer_ptrs(num_layers), bias_ptrs(num_layers);
	std::vector<torch::Tensor> out_layers(num_layers), out_biases(num_layers);
	std::vector<float*> out_layer_ptrs(num_layers), out_bias_ptrs(num_layers);

	std::vector<float> layer_sizes(num_layers), bias_sizes(num_layers);

	curandGenerator_t gen;
	float* d_agent_ids, d_Rs, d_ris;
	int num_agents = NP * 3, num_Rs = NP, num_ris = NP * num_layers;
	cudaMalloc(&d_agent_ids, num_agents * sizeof(float));
	cudaMalloc(&d_Rs, num_Rs * sizeof(float));
	cudaMalloc(&d_ris, num_ris * sizeof(float));
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
	curandGenerateUniform(gen, d_agent_ids, num_agents);
	curandGenerateUniform(gen, d_Rs, num_Rs);
	curandGenerateUniform(gen, d_ris, num_ris);

	for (int i = 0; i < num_layers; i++) {
		torch::Tensor layer_contig = layers[i].contiguous();
		torch::Tensor bias_contig = biases[i].contiguous();
		layer_ptrs[i] = layer_contig.data_ptr<float>();
		bias_ptrs[i] = bias_contig.data_ptr<float>();

		//out_layers[i] = torch::empty(layer_contig.sizes(), layer_contig.options());
		//out_biases[i] = torch::empty(bias_contig.sizes(), bias_contig.options());
		out_layers[i] = torch::clone(layer_contig);
		out_biases[i] = torch::clone(bias_contig);
		out_layer_ptrs[i] = out_layers[i].data_ptr<float>();
		out_bias_ptrs[i] = out_biases[i].data_ptr<float>();

		layer_sizes[i] = layer_contig.numel() / NP;
		bias_sizes[i] = bias_contig.numel() / NP;
		std::cout << "layer " << i << " has " << layer_sizes[i] << " parameters" << std::endl;
		std::cout << "bias  " << i << " has " << bias_sizes[i]  << " parameters" << std::endl;
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
	
	float* d_layer_sizes = nullptr;
	float* d_bias_sizes = nullptr;
	cudaMalloc(&d_layer_sizes, num_layers * sizeof(float));
	cudaMalloc(&d_bias_sizes, num_layers * sizeof(float));

	de_crossover_kernel<<<max(1l, NP / 64), 64>>>(NP, num_layers, CR * 0x1.0p32, F, best_model, d_layer_ptrs, d_bias_ptrs, d_out_layer_ptrs, d_out_bias_ptrs);

	cudaFree(d_layer_ptrs);
	cudaFree(d_bias_ptrs);
	cudaFree(d_out_layer_ptrs);
	cudaFree(d_out_bias_ptrs);

	return {out_layers, out_biases};
}
