#ifndef LOCALMODELS_H
#define LOCALMODELS_H

#include <Eigen/Dense>
#include <random>
#include <memory>
#include <iostream>
#include <torch/torch.h>

// Helper function for random matrix generation
Eigen::MatrixXd randomMatrix(int rows, int cols, double stddev = 0.1) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0, stddev);
    
    Eigen::MatrixXd m(rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m(i, j) = d(gen);
        }
    }
    return m;
}

// Helper functions for activations
Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x) {
    return (1.0 / (1.0 + (-x.array()).exp())).matrix();
}

Eigen::MatrixXd sigmoid_derivative(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd s = sigmoid(x);
    return (s.array() * (1.0 - s.array())).matrix();
}

Eigen::MatrixXd relu(const Eigen::MatrixXd& x) {
    return x.array().max(0.0).matrix();
}

Eigen::MatrixXd relu_derivative(const Eigen::MatrixXd& x) {
    return (x.array() > 0.0).cast<double>().matrix();
}

// Abstract base class for local models
class VFLBaseLocalModel {
public:
    virtual ~VFLBaseLocalModel() = default;
    virtual Eigen::MatrixXd forward(const Eigen::MatrixXd& x) const = 0;
    virtual void backward(const Eigen::MatrixXd& grad_output, double learning_rate) = 0;
    virtual void train() {}
    virtual void eval() {}
};

// ------------------------------
// Logistic Regression Local Model
// ------------------------------
class LRLocal : public VFLBaseLocalModel {
private:
    Eigen::MatrixXd W;
    Eigen::MatrixXd b;
    double learning_rate;
    mutable Eigen::MatrixXd last_input;
    
public:
    LRLocal(int input_dim, int output_dim, double lr = 0.01) 
        : learning_rate(lr) {
        W = randomMatrix(output_dim, input_dim);
        b = randomMatrix(output_dim, 1);
        
        std::cout << "LRLocal initialized with:" << std::endl;
        std::cout << "  W dimensions: " << W.rows() << "x" << W.cols() << std::endl;
        std::cout << "  b dimensions: " << b.rows() << "x" << b.cols() << std::endl;
    }
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) const override {
        std::cout << "LRLocal forward - Input dimensions: " << x.rows() << "x" << x.cols() << std::endl;
        
        if (x.rows() != W.cols()) {
            std::cerr << "Error in LRLocal forward: Input dimension mismatch. Expected " 
                      << W.cols() << " but got " << x.rows() << std::endl;
            return Eigen::MatrixXd::Zero(W.rows(), x.cols());
        }
        
        // Store input for backward pass
        last_input = x;
        
        // Apply linear transformation: y = W*x + b
        Eigen::MatrixXd output = W * x;
        
        // Add bias to each column
        for (int i = 0; i < x.cols(); ++i) {
            output.col(i) += b;
        }
        
        std::cout << "LRLocal output dimensions: " << output.rows() << "x" << output.cols() << std::endl;
        return output;
    }
    
    void backward(const Eigen::MatrixXd& grad_output, double learning_rate) override {
        // Compute weight gradients
        Eigen::MatrixXd dW = grad_output * last_input.transpose() / last_input.cols();
        
        // Compute bias gradients (average over batch)
        Eigen::MatrixXd db = grad_output.rowwise().mean();
        
        // Update weights and biases
        W -= learning_rate * dW;
        b -= learning_rate * db;
    }
    
    // Get model parameters
    std::vector<torch::Tensor> get_parameters() const {
        auto W_tensor = torch::from_blob(const_cast<double*>(W.data()), 
                                        {static_cast<long>(W.rows()), static_cast<long>(W.cols())},
                                        torch::kDouble).clone();
        auto b_tensor = torch::from_blob(const_cast<double*>(b.data()), 
                                        {static_cast<long>(b.rows()), static_cast<long>(b.cols())},
                                        torch::kDouble).clone();
        return {W_tensor, b_tensor};
    }
    
    // Set model parameters
    void set_parameters(const std::vector<torch::Tensor>& params) {
        if (params.size() != 2) return;
        
        auto W_tensor = params[0].to(torch::kDouble);
        auto b_tensor = params[1].to(torch::kDouble);
        
        W = Eigen::Map<Eigen::MatrixXd>(W_tensor.data_ptr<double>(), 
                                        W_tensor.size(0), W_tensor.size(1));
        b = Eigen::Map<Eigen::MatrixXd>(b_tensor.data_ptr<double>(),
                                        b_tensor.size(0), b_tensor.size(1));
    }
    
    void train() override {}
    void eval() override {}
};

// ------------------------------
// MLP Local Model
// ------------------------------
class MLP4Local : public VFLBaseLocalModel {
private:
    Eigen::MatrixXd W1, b1, W2, b2;
    double learning_rate;
    mutable Eigen::MatrixXd last_input;
    mutable Eigen::MatrixXd z1, a1;
    
public:
    MLP4Local(int input_dim, int hidden_dim, int output_dim, double lr = 0.01)
        : learning_rate(lr) {
        // Initialize weights and biases
        W1 = randomMatrix(hidden_dim, input_dim);
        b1 = randomMatrix(hidden_dim, 1);
        W2 = randomMatrix(output_dim, hidden_dim);
        b2 = randomMatrix(output_dim, 1);
        
        std::cout << "MLP4Local initialized with:" << std::endl;
        std::cout << "  W1 dimensions: " << W1.rows() << "x" << W1.cols() << std::endl;
        std::cout << "  b1 dimensions: " << b1.rows() << "x" << b1.cols() << std::endl;
        std::cout << "  W2 dimensions: " << W2.rows() << "x" << W2.cols() << std::endl;
        std::cout << "  b2 dimensions: " << b2.rows() << "x" << b2.cols() << std::endl;
    }
    
    Eigen::MatrixXd forward(const Eigen::MatrixXd& x) const override {
        std::cout << "MLP4Local forward - Input dimensions: " << x.rows() << "x" << x.cols() << std::endl;
        
        if (x.rows() != W1.cols()) {
            std::cerr << "Error in MLP4Local forward: Input dimension mismatch. Expected " 
                      << W1.cols() << " but got " << x.rows() << std::endl;
            return Eigen::MatrixXd::Zero(W2.rows(), x.cols());
        }
        
        // Store input for backward pass
        last_input = x;
        
        // First layer: z1 = W1*x + b1
        z1 = W1 * x;
        for (int i = 0; i < x.cols(); ++i) {
            z1.col(i) += b1;
        }
        
        // Apply ReLU activation: a1 = ReLU(z1)
        a1 = relu(z1);
        
        // Second layer: z2 = W2*a1 + b2
        Eigen::MatrixXd z2 = W2 * a1;
        for (int i = 0; i < a1.cols(); ++i) {
            z2.col(i) += b2;
        }
        
        std::cout << "MLP4Local output dimensions: " << z2.rows() << "x" << z2.cols() << std::endl;
        return z2;
    }
    
    void backward(const Eigen::MatrixXd& grad_output, double learning_rate) override {
        int batch_size = last_input.cols();
        
        // Compute gradients for output layer
        Eigen::MatrixXd dW2 = grad_output * a1.transpose() / batch_size;
        Eigen::MatrixXd db2 = grad_output.rowwise().mean();
        
        // Backpropagate gradient to hidden layer
        Eigen::MatrixXd grad_hidden = W2.transpose() * grad_output;
        
        // Apply ReLU gradient: grad_hidden * ReLU'(z1)
        Eigen::MatrixXd relu_grad = (z1.array() > 0.0).cast<double>();
        Eigen::MatrixXd grad_hidden_activated = grad_hidden.array() * relu_grad.array();
        
        // Compute gradients for hidden layer
        Eigen::MatrixXd dW1 = grad_hidden_activated * last_input.transpose() / batch_size;
        Eigen::MatrixXd db1 = grad_hidden_activated.rowwise().mean();
        
        // Update weights and biases
        W1 -= learning_rate * dW1;
        b1 -= learning_rate * db1;
        W2 -= learning_rate * dW2;
        b2 -= learning_rate * db2;
    }
    
    // Get model parameters
    std::vector<torch::Tensor> get_parameters() const {
        std::vector<torch::Tensor> params;
        params.push_back(torch::from_blob(const_cast<double*>(W1.data()), 
                                        {static_cast<long>(W1.rows()), static_cast<long>(W1.cols())},
                                        torch::kDouble).clone());
        params.push_back(torch::from_blob(const_cast<double*>(b1.data()), 
                                        {static_cast<long>(b1.rows()), static_cast<long>(b1.cols())},
                                        torch::kDouble).clone());
        params.push_back(torch::from_blob(const_cast<double*>(W2.data()), 
                                        {static_cast<long>(W2.rows()), static_cast<long>(W2.cols())},
                                        torch::kDouble).clone());
        params.push_back(torch::from_blob(const_cast<double*>(b2.data()), 
                                        {static_cast<long>(b2.rows()), static_cast<long>(b2.cols())},
                                        torch::kDouble).clone());
        return params;
    }
    
    // Set model parameters
    void set_parameters(const std::vector<torch::Tensor>& params) {
        if (params.size() != 4) return;
        
        auto W1_tensor = params[0].to(torch::kDouble);
        auto b1_tensor = params[1].to(torch::kDouble);
        auto W2_tensor = params[2].to(torch::kDouble);
        auto b2_tensor = params[3].to(torch::kDouble);
        
        W1 = Eigen::Map<Eigen::MatrixXd>(W1_tensor.data_ptr<double>(), 
                                        W1_tensor.size(0), W1_tensor.size(1));
        b1 = Eigen::Map<Eigen::MatrixXd>(b1_tensor.data_ptr<double>(),
                                        b1_tensor.size(0), b1_tensor.size(1));
        W2 = Eigen::Map<Eigen::MatrixXd>(W2_tensor.data_ptr<double>(),
                                        W2_tensor.size(0), W2_tensor.size(1));
        b2 = Eigen::Map<Eigen::MatrixXd>(b2_tensor.data_ptr<double>(),
                                        b2_tensor.size(0), b2_tensor.size(1));
    }
    
    void train() override {}
    void eval() override {}
};

#endif // LOCALMODELS_H

