import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.cuda.nvtx as nvtx
from torchvision import datasets
import matplotlib.pyplot as plt
import math
import random
import copy
import sys
import time
from IPython.display import clear_output
from torch.utils.cpp_extension import load_inline, load

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
torch.manual_seed(2244)

cuda_source = '''
#include <curand.h>

__global__ void de_crossover_kernel(int NP, float CR, float F, int best_model, float* d_ptr, float* d_out_ptr, int size, float* d_all_agent_ids, float* d_Rs, float* d_ris, int layer_idx, int num_layers) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < NP * size) {
		int id = idx / size; // candidate id
		int agent_ids[3]{d_all_agent_ids[id * 3 + 0] * NP, d_all_agent_ids[id * 3 + 1] * NP, d_all_agent_ids[id * 3 + 2] * NP};
		//printf("id: %d, best model: %d, agent 0: %d, agent 1: %d\\n", id, best_model, agent_ids[0], agent_ids[1]);
		int R = d_Rs[id] * (float)num_layers;
		float ri = d_ris[layer_idx * NP + id];

		if (ri < CR || layer_idx == R) {
			d_out_ptr[idx] = d_ptr[idx] + F * (d_ptr[best_model * size + idx % size] - d_ptr[idx]) + F * (d_ptr[agent_ids[0] * size + idx % size] - d_ptr[agent_ids[1] * size + idx % size]);
			printf("crossover layer %d of id %d with agent0 %d and agent1 %d using ri %f and R %d \\n id d_ptr[%d] = %f best_model d_ptr[%d] = %f agent0 d_ptr[%d] = %f agent1 d_ptr[%d] = %f\\n", layer_idx, id, agent_ids[0], agent_ids[1], ri, R,
				idx, d_ptr[idx], best_model * size + idx % size, d_ptr[best_model * size + idx % size], agent_ids[0] * size + idx % size, d_ptr[agent_ids[0] * size + idx % size], agent_ids[1] * size + idx % size, d_ptr[agent_ids[1] * size + idx % size]);
		}
	}
}

std::vector<std::vector<torch::Tensor>> de_crossover_cuda(const std::vector<torch::Tensor>& layers, const std::vector<torch::Tensor>& biases, int64_t NP, double CR, double F, int64_t best_model) {
	int num_layers = layers.size();
	std::vector<float*> layer_ptrs(num_layers), bias_ptrs(num_layers);
	std::vector<torch::Tensor> out_layers(num_layers), out_biases(num_layers);
	std::vector<float*> out_layer_ptrs(num_layers), out_bias_ptrs(num_layers);

	curandGenerator_t gen;
	float* d_all_agent_ids;
	float* d_Rs;
	float* d_ris;
	int num_agents = NP * 3, num_Rs = NP, num_ris = num_layers * NP;
	cudaMalloc(&d_all_agent_ids, num_agents * sizeof(float));
	cudaMalloc(&d_Rs, num_Rs * sizeof(float));
	cudaMalloc(&d_ris, num_ris * sizeof(float));
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 5691ULL);
	curandGenerateUniform(gen, d_all_agent_ids, num_agents);
	curandGenerateUniform(gen, d_Rs, num_Rs);
	curandGenerateUniform(gen, d_ris, num_ris);
	//std::cout << "num_layers " << num_layers << std::endl;

	for (int i = 0; i < num_layers; i++) {
		// I shouldnt be copying tensors here...
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

		de_crossover_kernel<<<max(1l, layer_contig.numel() / 64), 64>>>(NP, CR, F, best_model, layer_ptrs[i], out_layer_ptrs[i], layer_contig.numel() / NP, d_all_agent_ids, d_Rs, d_ris, i, num_layers);
		de_crossover_kernel<<<max(1l, bias_contig.numel() / 64), 64>>>(NP, CR, F, best_model, bias_ptrs[i], out_bias_ptrs[i], bias_contig.numel() / NP, d_all_agent_ids, d_Rs, d_ris, i, num_layers);
		
		//std::cout << "layer " << i << " has " << layer_contig.numel() / NP << " parameters" << std::endl;
		//std::cout << "bias  " << i << " has " << bias_contig.numel() / NP  << " parameters" << std::endl;
	}

	cudaFree(d_all_agent_ids);
	cudaFree(d_Rs);
	cudaFree(d_ris);

	return {out_layers, out_biases};
}
'''

cpp_source = '''
std::vector<std::vector<torch::Tensor>> de_crossover_cuda(const std::vector<torch::Tensor>& layers, const std::vector<torch::Tensor>& biases, int64_t NP, double CR, double F, int64_t best_model);
'''

# Load the CUDA kernel as a PyTorch extension
diff_evo = load_inline(
    name='diff_evo',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['de_crossover_cuda'],
    with_cuda=True,
    extra_cuda_cflags=["-O3", "-lcurand", "-L/usr/local/cuda-12.8/lib64"],
    build_directory='./diff_evo_cuda',
    # extra_cuda_cflags=['--expt-relaxed-constexpr']
)

NP = 5
lin1s = nn.init.kaiming_uniform_(torch.empty((NP, 1, 1), requires_grad=False).to(device, non_blocking=True))
lin2s = nn.init.kaiming_uniform_(torch.empty((NP, 2, 2), requires_grad=False).to(device, non_blocking=True))
layers = [lin1s, lin2s]
print(layers)

bias1 = nn.init.kaiming_uniform_(torch.empty((NP, 1, 1), requires_grad=False).to(device, non_blocking=True))
bias2 = nn.init.kaiming_uniform_(torch.empty((NP, 2, 1), requires_grad=False).to(device, non_blocking=True))
biases = [bias1, bias2]

lst = diff_evo.de_crossover_cuda(layers, biases, NP, 0.9, 0.8, 0)
print(lst[0])