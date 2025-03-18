#ifndef LOCALMODELS_H
#define LOCALMODELS_H

#include "VFLBaseLocalModel.h"

// ------------------------------
// Logistic Regression Local Model
// ------------------------------
struct LRLocalImpl : VFLBaseLocalModel {
    torch::nn::Sequential layer{nullptr};

    // Constructor: create a model that flattens input then applies a linear layer.
    LRLocalImpl(int64_t input_dim, int64_t output_dim) {
        layer = torch::nn::Sequential(
            torch::nn::Flatten(),
            torch::nn::Linear(input_dim, output_dim)
        );
        register_module("layer", layer);
    }

    // Forward pass: apply the sequential layer.
    torch::Tensor forward(torch::Tensor x) override {
        return layer->forward(x);
    }
};
TORCH_MODULE(LRLocal);

// ------------------------------
// MLP Local Model
// ------------------------------
struct MLP4LocalImpl : VFLBaseLocalModel {
    torch::nn::Sequential model{nullptr};

    // Constructor: build an MLP with multiple layers.
    MLP4LocalImpl(int64_t input_dim, int64_t output_dim) {
        model = torch::nn::Sequential(
            torch::nn::Linear(input_dim, 100),
            torch::nn::ReLU(),
            torch::nn::Linear(100, 50),
            torch::nn::ReLU(),
            torch::nn::Linear(50, 20),
            torch::nn::ReLU(),
            torch::nn::Linear(20, output_dim)
        );
        register_module("model", model);
    }

    // Forward pass: run the input through the MLP.
    torch::Tensor forward(torch::Tensor x) override {
        return model->forward(x);
    }
};
TORCH_MODULE(MLP4Local);

#endif // LOCALMODELS_H

