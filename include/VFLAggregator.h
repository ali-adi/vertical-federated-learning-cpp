#ifndef VFLAGGREGATOR_H
#define VFLAGGREGATOR_H

#include <torch/torch.h>

// Aggregator that collects outputs from local models, concatenates them, and applies a final linear layer.
struct VFLAggregatorImpl : torch::nn::Module {
    torch::nn::Linear final_layer{nullptr};

    // Constructor: aggregator_input_dim is the sum of local output dimensions.
    VFLAggregatorImpl(int64_t aggregator_input_dim, int64_t aggregator_output_dim) {
        final_layer = register_module(
            "final_layer",
            torch::nn::Linear(aggregator_input_dim, aggregator_output_dim)
        );
    }

    // Forward pass: concatenate local outputs along the feature dimension and produce a final output.
    torch::Tensor forward(const std::vector<torch::Tensor>& local_outputs) {
        torch::Tensor concat = torch::cat(local_outputs, /*dim=*/1);
        return final_layer->forward(concat);
    }
};

TORCH_MODULE(VFLAggregator);

#endif // VFLAGGREGATOR_H

