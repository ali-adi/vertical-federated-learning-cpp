#ifndef VFLBASELOCALMODEL_H
#define VFLBASELOCALMODEL_H

#include <torch/torch.h>

// Abstract base class for local models.
struct VFLBaseLocalModel : torch::nn::Module {
    // Pure virtual forward method that must be implemented by derived classes.
    virtual torch::Tensor forward(torch::Tensor x) = 0;
};

#endif // VFLBASELOCALMODEL_H

