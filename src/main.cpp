#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <cstdlib>
#include <Eigen/Dense>
#include <random>

#include "DataUtils.h"
#include "EvaluateUtils.h"
#include "LocalModels.h"
#include "VFLAggregator.h"
#include "MyDataset.h"

// Binary cross entropy loss function for Eigen matrices
double bce_loss(const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
    // Apply sigmoid to predictions
    Eigen::MatrixXd probs = sigmoid(y_pred);
    
    // Calculate BCE loss: -y_true * log(probs) - (1 - y_true) * log(1 - probs)
    double loss = 0.0;
    int batch_size = y_pred.cols();
    
    // Add small epsilon to avoid log(0)
    double epsilon = 1e-7;
    
    // Compute element-wise loss
    for (int i = 0; i < batch_size; ++i) {
        double y = y_true(0, i);
        double p = std::max(std::min(probs(0, i), 1.0 - epsilon), epsilon);
        loss += -y * std::log(p) - (1.0 - y) * std::log(1.0 - p);
    }
    
    // Return average loss
    return loss / batch_size;
}

// Gradient of binary cross entropy loss with respect to pre-sigmoid predictions
Eigen::MatrixXd bce_loss_grad(const Eigen::MatrixXd& y_pred, const Eigen::MatrixXd& y_true) {
    // Apply sigmoid to get probabilities
    Eigen::MatrixXd probs = sigmoid(y_pred);
    
    // Compute gradient: probs - y_true
    Eigen::MatrixXd grad = probs - y_true;
    
    return grad;
}

// Function to parse command line arguments
std::string getArgValue(int argc, char** argv, const std::string& arg, const std::string& defaultValue) {
    for (int i = 1; i < argc - 1; i++) {
        if (std::string(argv[i]) == arg) {
            return std::string(argv[i + 1]);
        }
    }
    return defaultValue;
}

// Structure to store dataset configuration
struct DatasetConfig {
    std::string csv_path;
    int num_csv_columns;
    int feature_start;
    int num_features_csv;
    int target_index;
};

int main(int argc, char** argv) {
    std::cout << "=== VFL Simulation Start ===" << std::endl;

    // Parse command line arguments
    std::string dataset_name = getArgValue(argc, argv, "--data", "credit");
    std::cout << "Using dataset: " << dataset_name << std::endl;

    // Configure dataset based on name
    DatasetConfig config;
    bool is_neurips = false;

    if (dataset_name == "credit") {
        config.csv_path = "../data/credit/default_of_credit_card_clients.csv";
        config.num_csv_columns = 25;
        config.feature_start = 1; // First column is ID, skip it
        config.num_features_csv = 23;
        config.target_index = 24; // Last column
    } else if (dataset_name == "credit-balanced") {
        config.csv_path = "../data/credit/default_of_credit_card_clients-balanced.csv";
        config.num_csv_columns = 25;
        config.feature_start = 1; // First column is ID, skip it
        config.num_features_csv = 23;
        config.target_index = 24; // Last column
    } else if (dataset_name.find("neurips") != std::string::npos) {
        is_neurips = true;
        // Determine which Neurips variant to use
        std::string variant;
        if (dataset_name == "neurips-base") {
            variant = "Base";
        } else if (dataset_name == "neurips-v1") {
            variant = "Variant I";
        } else if (dataset_name == "neurips-v2") {
            variant = "Variant II";
        } else if (dataset_name == "neurips-v3") {
            variant = "Variant III";
        } else if (dataset_name == "neurips-v4") {
            variant = "Variant IV";
        } else if (dataset_name == "neurips-v5") {
            variant = "Variant V";
        } else {
            // Default to Base
            variant = "Base";
        }
        
        config.csv_path = "../data/neurips/" + variant + ".csv";
        config.num_csv_columns = 32; // NeurIPS data has 32 columns
        config.target_index = 0;     // First column for Neurips is target (fraud_bool)
        config.feature_start = 1;    // Features start after target
        config.num_features_csv = config.num_csv_columns - 1; // All columns except target
    } else {
        std::cerr << "Unknown dataset: " << dataset_name << ". Using default (credit)." << std::endl;
        config.csv_path = "../data/credit/default_of_credit_card_clients.csv";
        config.num_csv_columns = 25;
        config.feature_start = 1;
        config.num_features_csv = 23;
        config.target_index = 24;
    }

    // ==================== Hyperparameter Block ====================
    const double train_ratio = 0.7;
    const double val_ratio = 0.15;
    const double test_ratio = 0.15;
    const int64_t batch_size = 128;
    // Split column is half of features for all datasets
    const int64_t split_col = config.num_features_csv / 2;
    const int64_t left_input_dim = split_col;
    const int64_t right_input_dim = config.num_features_csv - split_col;
    const int64_t left_output_dim = 64;
    const int64_t right_output_dim = 64;
    const int64_t aggregator_input_dim = left_output_dim + right_output_dim;
    const int64_t aggregator_output_dim = 1;
    const size_t num_epochs = 10;
    const double learning_rate = 0.001;
    // ================================================================

    // Create run folder (in project root)
    std::filesystem::create_directories("../runs");
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm *now_tm = std::localtime(&now_time);
    std::ostringstream oss;
    oss << std::put_time(now_tm, "%Y%m%d_%H%M%S");
    std::string run_folder_name = oss.str();
    std::string run_folder_path = "../runs/" + run_folder_name;
    std::filesystem::create_directories(run_folder_path);

    // ------------------------------
    // 1. Load CSV data.
    // ------------------------------
    std::cout << "Loading data from: " << config.csv_path << std::endl;
    auto csv_data = loadCSV(config.csv_path);
    if (csv_data.empty()) {
        std::cerr << "ERROR: No data loaded from CSV." << std::endl;
        return -1;
    }

    // ------------------------------
    // 2. Parse CSV data into features and labels.
    // ------------------------------
    std::vector<std::vector<float>> feature_rows;
    std::vector<float> labels;
    
    for (const auto &row : csv_data) {
        if (row.size() != config.num_csv_columns) {
            std::cerr << "Warning: Skipping row with incorrect column count. Expected " 
                      << config.num_csv_columns << ", got " << row.size() << std::endl;
            continue;
        }
        
        std::vector<float> features;
        
        // For neurips datasets, extract target (first column) and then features
        if (is_neurips) {
            try { 
                labels.push_back(std::stof(row[config.target_index])); 
            } catch (...) { 
                labels.push_back(0.0f); 
            }
            
            for (int j = config.feature_start; j < config.num_csv_columns; j++) {
                try { features.push_back(std::stof(row[j])); }
                catch (...) { features.push_back(0.0f); }
            }
        } 
        // For credit datasets, extract features and then target (last column)
        else {
            for (int j = config.feature_start; j < config.feature_start + config.num_features_csv; j++) {
                try { features.push_back(std::stof(row[j])); }
                catch (...) { features.push_back(0.0f); }
            }
            
            try { labels.push_back(std::stof(row[config.target_index])); }
            catch (...) { labels.push_back(0.0f); }
        }
        
        feature_rows.push_back(features);
    }

    // *** Shuffle the data so that splits are random ***
    shuffleData(feature_rows, labels);

    // Convert to Eigen matrices
    auto[features_matrix, labels_matrix] = convertToEigenMatrices(feature_rows, labels);
    long num_rows = features_matrix.cols(); // Number of samples
    
    std::cout << "Features matrix dimensions: " << features_matrix.rows() << "x" << features_matrix.cols() << std::endl;
    std::cout << "Labels matrix dimensions: " << labels_matrix.rows() << "x" << labels_matrix.cols() << std::endl;
    
    // ------------------------------
    // 3. Split data into train/val/test sets.
    // ------------------------------
    int64_t train_size = static_cast<int64_t>(train_ratio * num_rows);
    int64_t val_size = static_cast<int64_t>(val_ratio * num_rows);
    int64_t test_size = num_rows - train_size - val_size;
    
    Eigen::MatrixXd train_features = features_matrix.block(0, 0, features_matrix.rows(), train_size);
    Eigen::MatrixXd train_labels = labels_matrix.block(0, 0, 1, train_size);
    Eigen::MatrixXd val_features = features_matrix.block(0, train_size, features_matrix.rows(), val_size);
    Eigen::MatrixXd val_labels = labels_matrix.block(0, train_size, 1, val_size);
    Eigen::MatrixXd test_features = features_matrix.block(0, train_size + val_size, features_matrix.rows(), test_size);
    Eigen::MatrixXd test_labels = labels_matrix.block(0, train_size + val_size, 1, test_size);

    // ------------------------------
    // 4. Partition features for left/right models.
    // ------------------------------
    auto [train_features_left, train_features_right] = verticalSplit(train_features, split_col);
    auto [val_features_left, val_features_right] = verticalSplit(val_features, split_col);
    auto [test_features_left, test_features_right] = verticalSplit(test_features, split_col);

    // ------------------------------
    // 5. Create datasets and dataloaders.
    // ------------------------------
    EigenDataset train_dataset(train_features, train_labels);
    EigenDataset val_dataset(val_features, val_labels);
    EigenDataset test_dataset(test_features, test_labels);
    
    EigenDataLoader train_loader(train_dataset, batch_size, true);
    EigenDataLoader val_loader(val_dataset, batch_size, false);
    EigenDataLoader test_loader(test_dataset, batch_size, false);

    // ------------------------------
    // 6. Initialize models and aggregator.
    // ------------------------------
    LRLocal lrLocalModel(left_input_dim, left_output_dim);
    int hidden_dim = 100;  // Add hidden dimension for MLP4Local
    MLP4Local mlpLocalModel(right_input_dim, hidden_dim, right_output_dim);
    std::vector<int> local_output_dims = {left_output_dim, right_output_dim};
    VFLAggregator vflAggregator(local_output_dims, aggregator_output_dim);

    // Vectors to record epoch losses.
    std::vector<double> epoch_train_losses;
    std::vector<double> epoch_val_losses;
    double best_val_accuracy = 0.0;

    // ------------------------------
    // 8. Training loop with checkpointing.
    // ------------------------------
    std::cout << "Training for " << num_epochs << " epochs..." << std::endl;
    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        double running_loss = 0.0;
        size_t total_samples = 0;
        
        for (const auto &batch : train_loader) {
            auto inputs = batch.data;
            auto targets = batch.target;
            size_t bsize = inputs.cols();
            total_samples += bsize;
            
            // Split inputs for local models
            auto [inputs_left, inputs_right] = verticalSplit(inputs, split_col);
            
            std::cout << "Batch dimensions: " << inputs.rows() << "x" << inputs.cols() << std::endl;
            std::cout << "Left split dimensions: " << inputs_left.rows() << "x" << inputs_left.cols() << std::endl;
            std::cout << "Right split dimensions: " << inputs_right.rows() << "x" << inputs_right.cols() << std::endl;
            
            // Forward pass
            auto local_out_left = lrLocalModel.forward(inputs_left);
            auto local_out_right = mlpLocalModel.forward(inputs_right);
            auto final_out = vflAggregator.forward({local_out_left, local_out_right});
            
            // Print dimensions before sigmoid
            std::cout << "Before sigmoid - Output dimensions: " << final_out.rows() << "x" << final_out.cols() << std::endl;
            
            // Apply sigmoid activation for binary classification
            Eigen::MatrixXd final_probs = sigmoid(final_out);
            
            // Print dimensions after sigmoid
            std::cout << "After sigmoid - Probs dimensions: " << final_probs.rows() << "x" << final_probs.cols() << std::endl;
            std::cout << "Target dimensions: " << targets.rows() << "x" << targets.cols() << std::endl;
            
            // Compute loss
            double loss = bce_loss(final_probs, targets);
            running_loss += loss * bsize;
            
            // Print dimensions for backward pass
            std::cout << "Before creating grad_out" << std::endl;
            
            // Backward pass
            auto grad_out = (final_probs - targets).array() * final_probs.array() * (1.0 - final_probs.array());
            
            std::cout << "Grad output dimensions: " << grad_out.rows() << "x" << grad_out.cols() << std::endl;
            
            // Backward pass through aggregator and get gradients for local models
            auto local_grads = vflAggregator.backward(grad_out, {local_out_left, local_out_right});
            
            // Backward pass through local models with their respective gradients
            lrLocalModel.backward(local_grads[0], learning_rate);
            mlpLocalModel.backward(local_grads[1], learning_rate);
        }
        
        double epoch_loss = running_loss / total_samples;
        epoch_train_losses.push_back(epoch_loss);
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                  << " - Training Loss: " << epoch_loss << std::endl;

        // Validation loop.
        double val_loss_sum = 0.0;
        size_t val_samples = 0;
        
        for (const auto &batch : val_loader) {
            auto inputs = batch.data;
            auto targets = batch.target;
            size_t bsize = inputs.cols();
            val_samples += bsize;
            
            // Split inputs for local models
            auto [inputs_left, inputs_right] = verticalSplit(inputs, split_col);
            
            // Forward pass (no gradient tracking needed)
            auto local_out_left = lrLocalModel.forward(inputs_left);
            auto local_out_right = mlpLocalModel.forward(inputs_right);
            auto final_out = vflAggregator.forward({local_out_left, local_out_right});
            
            // Apply sigmoid and compute loss
            Eigen::MatrixXd final_probs = sigmoid(final_out);
            double loss = bce_loss(final_probs, targets);
            val_loss_sum += loss * bsize;
        }
        
        double epoch_val_loss = val_loss_sum / val_samples;
        epoch_val_losses.push_back(epoch_val_loss);
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                  << " - Validation Loss: " << epoch_val_loss << std::endl;

        // Compute validation accuracy.
        double current_val_accuracy = compute_accuracy(lrLocalModel, mlpLocalModel, vflAggregator, val_loader, split_col);
        std::cout << "Epoch " << (epoch + 1) << " - Validation Accuracy: " << current_val_accuracy * 100 << "%" << std::endl;
        
        if (current_val_accuracy > best_val_accuracy) {
            best_val_accuracy = current_val_accuracy;
            
            // Save models for torch compatibility
            std::vector<torch::Tensor> lr_params = lrLocalModel.get_parameters();
            std::vector<torch::Tensor> mlp_params = mlpLocalModel.get_parameters();
            std::vector<torch::Tensor> agg_params = vflAggregator.get_parameters();
            
            torch::save(lr_params, run_folder_path + "/best_lrLocalModel.pt");
            torch::save(mlp_params, run_folder_path + "/best_mlpLocalModel.pt");
            torch::save(agg_params, run_folder_path + "/best_vflAggregator.pt");
            
            std::cout << "Checkpoint: Best model updated at epoch " << (epoch + 1) << std::endl;
        }
    }

    // ------------------------------
    // 9. Load best model for testing.
    // ------------------------------
    std::cout << "\nLoading best model for testing (Validation Accuracy: " << best_val_accuracy * 100 << "%)..." << std::endl;
    
    std::vector<torch::Tensor> lr_params;
    std::vector<torch::Tensor> mlp_params;
    std::vector<torch::Tensor> agg_params;
    
    torch::load(lr_params, run_folder_path + "/best_lrLocalModel.pt");
    torch::load(mlp_params, run_folder_path + "/best_mlpLocalModel.pt");
    torch::load(agg_params, run_folder_path + "/best_vflAggregator.pt");
    
    lrLocalModel.set_parameters(lr_params);
    mlpLocalModel.set_parameters(mlp_params);
    vflAggregator.set_parameters(agg_params);

    // ------------------------------
    // 10. Testing loop.
    // ------------------------------
    double test_loss_sum = 0.0;
    size_t test_samples = 0;
    
    for (const auto &batch : test_loader) {
        auto inputs = batch.data;
        auto targets = batch.target;
        size_t bsize = inputs.cols();
        test_samples += bsize;
        
        // Split inputs for local models
        auto [inputs_left, inputs_right] = verticalSplit(inputs, split_col);
        
        // Forward pass
        auto local_out_left = lrLocalModel.forward(inputs_left);
        auto local_out_right = mlpLocalModel.forward(inputs_right);
        auto final_out = vflAggregator.forward({local_out_left, local_out_right});
        
        // Apply sigmoid and compute loss
        Eigen::MatrixXd final_probs = sigmoid(final_out);
        double loss = bce_loss(final_probs, targets);
        test_loss_sum += loss * bsize;
    }
    
    double test_loss = test_loss_sum / test_samples;
    std::cout << "Test Loss: " << test_loss << std::endl;

    // ------------------------------
    // 11. Compute accuracies.
    // ------------------------------
    double train_accuracy = compute_accuracy(lrLocalModel, mlpLocalModel, vflAggregator, train_loader, split_col);
    double val_accuracy = compute_accuracy(lrLocalModel, mlpLocalModel, vflAggregator, val_loader, split_col);
    double test_accuracy = compute_accuracy(lrLocalModel, mlpLocalModel, vflAggregator, test_loader, split_col);

    std::cout << "Final Metrics:" << std::endl;
    std::cout << "Train Accuracy: " << train_accuracy * 100 << "%" << std::endl;
    std::cout << "Validation Accuracy: " << val_accuracy * 100 << "%" << std::endl;
    std::cout << "Test Accuracy: " << test_accuracy * 100 << "%" << std::endl;
    std::cout << "Test Loss: " << test_loss << std::endl;

    // ------------------------------
    // 12. Save loss curves and run details.
    // ------------------------------
    std::ofstream loss_file(run_folder_path + "/losses.csv");
    if (loss_file.is_open()) {
        loss_file << "epoch,train_loss,val_loss\n";
        for (size_t i = 0; i < epoch_train_losses.size(); i++) {
            loss_file << (i + 1) << "," << epoch_train_losses[i] << "," << epoch_val_losses[i] << "\n";
        }
        loss_file.close();
    } else {
        std::cerr << "ERROR: Could not open loss file for writing." << std::endl;
    }

    std::string command = "python3 ../python/plot_losses.py " + run_folder_path + "/losses.csv " +
                          run_folder_path + "/train_loss_plot.jpg " + run_folder_path + "/val_loss_plot.jpg";
    int ret = system(command.c_str());
    if (ret != 0) {
        std::cerr << "WARNING: Plotting script returned an error." << std::endl;
    }

    std::ofstream log_file(run_folder_path + "/run_details.txt");
    if (log_file.is_open()) {
        log_file << "==================== RUN DETAILS ====================\n\n";
        log_file << "Hyperparameters:\n";
        log_file << "------------------------------------------------------\n";
        log_file << "csv_path: " << config.csv_path << "\n";
        log_file << "num_csv_columns: " << config.num_csv_columns << "\n";
        log_file << "feature_start: " << config.feature_start << "\n";
        log_file << "num_features_csv: " << config.num_features_csv << "\n";
        log_file << "target_index: " << config.target_index << "\n";
        log_file << "train_ratio: " << train_ratio << "\n";
        log_file << "val_ratio: " << val_ratio << "\n";
        log_file << "test_ratio: " << test_ratio << "\n";
        log_file << "batch_size: " << batch_size << "\n";
        log_file << "split_col (left branch features): " << split_col << "\n";
        log_file << "left_input_dim: " << left_input_dim << "\n";
        log_file << "right_input_dim: " << right_input_dim << "\n";
        log_file << "left_output_dim: " << left_output_dim << "\n";
        log_file << "right_output_dim: " << right_output_dim << "\n";
        log_file << "aggregator_input_dim: " << aggregator_input_dim << "\n";
        log_file << "aggregator_output_dim: " << aggregator_output_dim << "\n";
        log_file << "num_epochs: " << num_epochs << "\n";
        log_file << "learning_rate: " << learning_rate << "\n\n";
        log_file << "Final Metrics:\n";
        log_file << "------------------------------------------------------\n";
        log_file << "Train Accuracy: " << train_accuracy * 100 << "%\n";
        log_file << "Validation Accuracy: " << val_accuracy * 100 << "%\n";
        log_file << "Test Accuracy: " << test_accuracy * 100 << "%\n";
        log_file << "Test Loss: " << test_loss << "\n";
        log_file << "Best Validation Accuracy: " << best_val_accuracy * 100 << "%\n";
        log_file << "======================================================\n";
        log_file.close();
    } else {
        std::cerr << "ERROR: Could not open log file for writing." << std::endl;
    }
    
    std::cout << "Training complete! Best model (Validation Accuracy: " << best_val_accuracy * 100 
              << "%) and run details saved in: " << run_folder_path << std::endl;
    return 0;
}
