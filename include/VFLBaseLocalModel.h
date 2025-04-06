#ifndef VFLBASELOCALMODEL_H
#define VFLBASELOCALMODEL_H

#include <Eigen/Dense>
#include <torch/torch.h> // Only kept for model saving/loading

// Abstract base class for local models.
class VFLBaseLocalModel {
public:
    virtual ~VFLBaseLocalModel() = default;
    
    // Pure virtual forward method that must be implemented by derived classes
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& x) = 0;

    // Virtual method to get parameters for saving
    virtual std::vector<torch::Tensor> get_parameters() const = 0;
    
    // Virtual method to set parameters from loaded model
    virtual void set_parameters(const std::vector<torch::Tensor>& params) = 0;
};

#endif // VFLBASELOCALMODEL_H

