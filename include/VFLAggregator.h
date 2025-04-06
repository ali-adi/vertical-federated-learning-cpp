#ifndef VFLAGGREGATOR_H
#define VFLAGGREGATOR_H

#include <Eigen/Dense>
#include <torch/torch.h> // Only kept for model saving/loading
#include "LocalModels.h" // For randomMatrix function
#include <vector>
#include <iostream>
#include <random>

// Aggregator that collects outputs from local models, concatenates them, and applies a final linear layer.
class VFLAggregator {
private:
    Eigen::MatrixXd W;  // Weight matrix (output_dim x input_dim)
    Eigen::MatrixXd b;  // Bias vector (output_dim x 1)
    double learning_rate;
    std::vector<int> local_dims; // Dimensions of each local model's output
    int total_local_dims; // Sum of all local dimensions

public:
    // Constructor: aggregator_input_dim is the sum of local output dimensions.
    VFLAggregator(const std::vector<int>& local_output_dims, int aggregator_output_dim, double lr = 0.01)
        : local_dims(local_output_dims), learning_rate(lr) {
        
        // Calculate total input dimension as sum of all local model output dimensions
        total_local_dims = 0;
        for (int dim : local_output_dims) {
            total_local_dims += dim;
        }
        
        // Initialize weight matrix with random values (output_dim x input_dim)
        W = randomMatrix(aggregator_output_dim, total_local_dims);
        
        // Initialize bias vector
        b = Eigen::MatrixXd::Zero(aggregator_output_dim, 1);
        
        std::cout << "VFLAggregator initialized with:" << std::endl;
        std::cout << "  W dimensions: " << W.rows() << "x" << W.cols() << std::endl;
        std::cout << "  b dimensions: " << b.rows() << "x" << b.cols() << std::endl;
        std::cout << "  Total local dims: " << total_local_dims << std::endl;
        for (size_t i = 0; i < local_dims.size(); ++i) {
            std::cout << "  Local dim " << i << ": " << local_dims[i] << std::endl;
        }
    }

    // Forward pass: concatenate local outputs and produce a final output.
    Eigen::MatrixXd forward(const std::vector<Eigen::MatrixXd>& local_outputs) const {
        // Print input dimensions for debugging
        std::cout << "VFLAggregator forward - Input dimensions:" << std::endl;
        for (size_t i = 0; i < local_outputs.size(); ++i) {
            std::cout << "Local output " << i << ": " << local_outputs[i].rows() << "x" << local_outputs[i].cols() << std::endl;
        }

        // Safety check - ensure there are local outputs to work with
        if (local_outputs.empty()) {
            std::cerr << "Error: No local outputs provided to forward pass" << std::endl;
            return Eigen::MatrixXd::Zero(W.rows(), local_outputs[0].cols());
        }
        
        // Check that we have the correct number of local outputs
        if (local_outputs.size() != local_dims.size()) {
            std::cerr << "Error: Expected " << local_dims.size() << " local outputs, got " 
                      << local_outputs.size() << std::endl;
            return Eigen::MatrixXd::Zero(W.rows(), local_outputs[0].cols());
        }
        
        // Get batch size from the first local output
        int batch_size = local_outputs[0].cols();
        
        // Safety check - ensure batch_size is valid
        if (batch_size <= 0) {
            std::cerr << "Error: Invalid batch size in forward pass: " << batch_size << std::endl;
            return Eigen::MatrixXd::Zero(W.rows(), batch_size);
        }
        
        // Concatenate all local outputs along rows (vertical concatenation)
        // This creates a matrix of size (total_local_dims x batch_size)
        int current_row = 0;
        Eigen::MatrixXd concatenated = Eigen::MatrixXd::Zero(total_local_dims, batch_size);
        
        for (size_t i = 0; i < local_outputs.size(); ++i) {
            if (local_outputs[i].cols() != batch_size) {
                std::cerr << "Error: Inconsistent batch sizes in local outputs" << std::endl;
                return Eigen::MatrixXd::Zero(W.rows(), batch_size);
            }
            
            // Copy this local output into the appropriate rows of the concatenated matrix
            concatenated.block(current_row, 0, local_dims[i], batch_size) = local_outputs[i];
            current_row += local_dims[i];
        }
        
        std::cout << "Concatenated dimensions: " << concatenated.rows() << "x" << concatenated.cols() << std::endl;
        
        // Apply W and b: output = W * concatenated + b (for each column in batch)
        Eigen::MatrixXd output = W * concatenated;
        
        // Add bias to each column
        for (int i = 0; i < batch_size; ++i) {
            output.col(i) += b;
        }
        
        std::cout << "Output dimensions: " << output.rows() << "x" << output.cols() << std::endl;
        
        return output;
    }
    
    // Backward pass computes gradients and updates parameters
    std::vector<Eigen::MatrixXd> backward(const Eigen::MatrixXd& grad_output, 
                                        const std::vector<Eigen::MatrixXd>& local_outputs) {
        std::cout << "Backward pass:" << std::endl;
        std::cout << "  Grad output dimensions: " << grad_output.rows() << "x" << grad_output.cols() << std::endl;
        
        try {
            // Sanity checks
            if (grad_output.rows() != W.rows()) {
                std::cerr << "Error: grad_output rows (" << grad_output.rows() 
                          << ") doesn't match W rows (" << W.rows() << ")" << std::endl;
                return {};
            }
            
            if (local_outputs.empty()) {
                std::cerr << "Error: No local outputs provided to backward pass" << std::endl;
                return {};
            }
            
            int batch_size = grad_output.cols();
            
            if (batch_size <= 0) {
                std::cerr << "Error: Invalid batch size in backward pass: " << batch_size << std::endl;
                return {};
            }
            
            // Check that all local outputs have the same batch size
            for (const auto& local_out : local_outputs) {
                if (local_out.cols() != batch_size) {
                    std::cerr << "Error: Inconsistent batch sizes in backward" << std::endl;
                    return {};
                }
            }
            
            // Concatenate local outputs
            int current_row = 0;
            Eigen::MatrixXd concatenated = Eigen::MatrixXd::Zero(total_local_dims, batch_size);
            
            for (size_t i = 0; i < local_outputs.size(); ++i) {
                if (i >= local_dims.size()) {
                    std::cerr << "Error: More local outputs than expected" << std::endl;
                    return {};
                }
                
                int local_dim = local_dims[i];
                int local_cols = local_outputs[i].cols();
                
                // Safety check for dimensions
                if (current_row + local_dim > concatenated.rows() || local_cols > concatenated.cols()) {
                    std::cerr << "Error: Dimension mismatch in concatenation" << std::endl;
                    std::cerr << "  Current row: " << current_row << ", Local dim: " << local_dim << std::endl;
                    std::cerr << "  Concatenated rows: " << concatenated.rows() << ", Local cols: " << local_cols << std::endl;
                    return {};
                }
                
                concatenated.block(current_row, 0, local_dim, local_cols) = local_outputs[i];
                current_row += local_dim;
            }
            
            // Compute gradient for W: dW = grad_output * concatenated^T / batch_size
            Eigen::MatrixXd dW = grad_output * concatenated.transpose() / batch_size;
            
            // Compute gradient for bias: db = mean of grad_output across batch
            Eigen::MatrixXd db = grad_output.rowwise().mean();
            
            // Compute gradient for concatenated local outputs: dL = W^T * grad_output
            Eigen::MatrixXd dConcatenated = W.transpose() * grad_output;
            
            std::cout << "  dConcatenated dimensions: " << dConcatenated.rows() << "x" << dConcatenated.cols() << std::endl;
            
            // Split the gradient for each local model
            std::vector<Eigen::MatrixXd> local_grads;
            current_row = 0;
            
            for (size_t i = 0; i < local_dims.size(); ++i) {
                int local_dim = local_dims[i];
                
                // Safety check for dimensions
                if (current_row + local_dim > dConcatenated.rows()) {
                    std::cerr << "Error: Dimension mismatch in gradient splitting" << std::endl;
                    std::cerr << "  Current row: " << current_row << ", Local dim: " << local_dim << std::endl;
                    std::cerr << "  dConcatenated rows: " << dConcatenated.rows() << std::endl;
                    return {};
                }
                
                Eigen::MatrixXd local_grad = dConcatenated.block(current_row, 0, local_dim, batch_size);
                local_grads.push_back(local_grad);
                current_row += local_dim;
                
                std::cout << "  Local grad " << i << " dimensions: " << local_grad.rows() << "x" 
                          << local_grad.cols() << std::endl;
            }
            
            // Update parameters
            W -= learning_rate * dW;
            b -= learning_rate * db;
            
            return local_grads;
        } catch (const std::exception& e) {
            std::cerr << "Exception in VFLAggregator::backward: " << e.what() << std::endl;
            return {};
        }
    }
    
    std::vector<torch::Tensor> get_parameters() const {
        // Convert Eigen matrices to torch tensors for saving
        auto W_tensor = torch::from_blob(const_cast<double*>(W.data()), 
                                        {static_cast<long>(W.rows()), static_cast<long>(W.cols())},
                                        torch::kDouble).clone();
        auto b_tensor = torch::from_blob(const_cast<double*>(b.data()), 
                                        {static_cast<long>(b.rows()), static_cast<long>(b.cols())},
                                        torch::kDouble).clone();
        return {W_tensor, b_tensor};
    }
    
    void set_parameters(const std::vector<torch::Tensor>& params) {
        if (params.size() != 2) return;
        
        auto W_tensor = params[0].to(torch::kDouble);
        auto b_tensor = params[1].to(torch::kDouble);
        
        W = Eigen::Map<Eigen::MatrixXd>(W_tensor.data_ptr<double>(), 
                                        W_tensor.size(0), W_tensor.size(1));
        b = Eigen::Map<Eigen::MatrixXd>(b_tensor.data_ptr<double>(),
                                        b_tensor.size(0), b_tensor.size(1));
    }
    
    void train() {}
    void eval() {}

    // Get model parameters as matrices
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd> getParameters() const {
        return {W, b};
    }
    
    // Set model parameters from matrices
    void setParameters(const Eigen::MatrixXd& weights, const Eigen::MatrixXd& bias) {
        if (weights.rows() != W.rows() || weights.cols() != W.cols()) {
            std::cerr << "Error: Weight dimensions don't match" << std::endl;
            return;
        }
        
        if (bias.rows() != b.rows() || bias.cols() != b.cols()) {
            std::cerr << "Error: Bias dimensions don't match" << std::endl;
            return;
        }
        
        W = weights;
        b = bias;
    }
};

#endif // VFLAGGREGATOR_H

