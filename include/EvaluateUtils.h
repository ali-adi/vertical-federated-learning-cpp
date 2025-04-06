#ifndef EVALUATEUTILS_H
#define EVALUATEUTILS_H

#include <Eigen/Dense>
#include <vector>
#include <functional>
#include "LocalModels.h"
#include "VFLAggregator.h"
#include "MyDataset.h"
#include <iostream>
#include <cmath>

// Function to compute binary classification accuracy for an EigenDataLoader
double compute_accuracy(EigenDataLoader& data_loader, std::function<Eigen::MatrixXd(const Eigen::MatrixXd&)> model_forward) {
    try {
        size_t correct = 0;
        size_t total = 0;
        size_t batch_size = 0;
        size_t total_batches = data_loader.num_batches();
        size_t current_batch = 0;

        for (const auto& batch : data_loader) {
            try {
                current_batch++;
                
                std::cout << "Processing batch " << current_batch << "/" << total_batches 
                          << " with dimensions: " << batch.data.rows() << "x" << batch.data.cols() << std::endl;
                
                if (batch.data.cols() == 0) {
                    std::cerr << "Warning: Empty batch encountered, skipping" << std::endl;
                    continue;
                }
                
                // Forward pass
                Eigen::MatrixXd output = model_forward(batch.data);
                
                std::cout << "Output dimensions: " << output.rows() << "x" << output.cols() << std::endl;
                std::cout << "Target dimensions: " << batch.target.rows() << "x" << batch.target.cols() << std::endl;
                
                // Apply sigmoid to get probabilities
                Eigen::MatrixXd probabilities = sigmoid(output);
                
                if (probabilities.rows() == 0 || probabilities.cols() == 0) {
                    std::cerr << "Warning: Invalid probabilities dimensions: " 
                              << probabilities.rows() << "x" << probabilities.cols() << std::endl;
                    continue;
                }
                
                // Convert to binary predictions (threshold at 0.5)
                Eigen::MatrixXd predictions = (probabilities.array() > 0.5).cast<double>();
                
                // Count correct predictions
                batch_size = batch.data.cols();
                for (int i = 0; i < batch_size; ++i) {
                    if (i < predictions.cols() && i < batch.target.cols()) {
                        bool pred_correct = (predictions(0, i) == batch.target(0, i));
                        if (pred_correct) {
                            correct++;
                        }
                        total++;
                    }
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Exception during batch processing: " << e.what() << std::endl;
                continue;
            }
        }
        
        std::cout << "Accuracy calculation complete. Correct: " << correct << ", Total: " << total << std::endl;
        
        if (total == 0) {
            std::cerr << "Warning: No samples processed for accuracy calculation" << std::endl;
            return 0.0;
        }
        
        return static_cast<double>(correct) / total;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in compute_accuracy: " << e.what() << std::endl;
        return 0.0;
    }
}

// Compute binary accuracy for VFL model
double compute_accuracy(const LRLocal& lr_model, 
                       const MLP4Local& mlp_model,
                       const VFLAggregator& aggregator,
                       EigenDataLoader& data_loader,
                       int split_col) {
    try {
        size_t correct = 0;
        size_t total = 0;
        
        std::cout << "Computing accuracy on " << data_loader.num_batches() << " batches" << std::endl;
        
        for (const auto& batch : data_loader) {
            try {
                auto inputs = batch.data;
                auto targets = batch.target;
                
                if (inputs.cols() == 0) {
                    std::cerr << "Empty batch, skipping" << std::endl;
                    continue;
                }
                
                std::cout << "Batch size: " << inputs.cols() << std::endl;
                
                // Split inputs for local models
                auto [inputs_left, inputs_right] = verticalSplit(inputs, split_col);
                
                // Forward pass through local models
                auto local_out_left = lr_model.forward(inputs_left);
                auto local_out_right = mlp_model.forward(inputs_right);
                
                // Forward pass through aggregator
                auto final_out = aggregator.forward({local_out_left, local_out_right});
                
                // Apply sigmoid and threshold
                Eigen::MatrixXd probs = sigmoid(final_out);
                Eigen::MatrixXd predictions = (probs.array() > 0.5).cast<double>();
                
                // Count correct predictions
                for (int i = 0; i < targets.cols() && i < predictions.cols(); ++i) {
                    if (predictions(0, i) == targets(0, i)) {
                        correct++;
                    }
                    total++;
                }
            } catch (const std::exception& e) {
                std::cerr << "Exception in compute_accuracy batch processing: " << e.what() << std::endl;
                continue;
            }
        }
        
        std::cout << "Accuracy calculation complete. Correct: " << correct << ", Total: " << total << std::endl;
        
        if (total == 0) {
            std::cerr << "No samples processed for accuracy calculation" << std::endl;
            return 0.0;
        }
        
        return static_cast<double>(correct) / total;
    } catch (const std::exception& e) {
        std::cerr << "Exception in compute_accuracy: " << e.what() << std::endl;
        return 0.0;
    }
}

#endif // EVALUATEUTILS_H

