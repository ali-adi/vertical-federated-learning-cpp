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

#include "DataUtils.h"
#include "EvaluateUtils.h"
#include "TrainUtils.h"
#include "LocalModels.h"
#include "VFLAggregator.h"
#include "MyDataset.h"

int main() {
    std::cout << "=== VFL Simulation Start ===" << std::endl;

    // ==================== Hyperparameter Block ====================
    const std::string csv_path = "/Users/ali/root/University/Y4S2/DTC/VFL_CPP_Project/data/default_of_credit_card_clients-balanced.csv";
    const int num_csv_columns = 25;
    const int feature_start = 1;
    const int num_features_csv = 23;
    const int target_index = 24;
    const double train_ratio = 0.7;
    const double val_ratio = 0.15;
    const double test_ratio = 0.15;
    const int64_t batch_size = 128;
    const int64_t split_col = 6;
    const int64_t left_input_dim = split_col;
    const int64_t right_input_dim = num_features_csv - split_col;
    const int64_t left_output_dim = 64;
    const int64_t right_output_dim = 64;
    const int64_t aggregator_input_dim = left_output_dim + right_output_dim;
    const int64_t aggregator_output_dim = 1;
    const size_t num_epochs = 600;
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
    auto csv_data = loadCSV(csv_path);
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
        if (row.size() != num_csv_columns) continue;
        std::vector<float> features;
        for (int j = feature_start; j < feature_start + num_features_csv; j++) {
            try { features.push_back(std::stof(row[j])); }
            catch (...) { features.push_back(0.0f); }
        }
        feature_rows.push_back(features);
        try { labels.push_back(std::stof(row[target_index])); }
        catch (...) { labels.push_back(0.0f); }
    }

    // *** New: Shuffle the data so that splits are random ***
    shuffleData(feature_rows, labels);

    auto data_options = torch::TensorOptions().dtype(torch::kFloat32);
    long num_rows = feature_rows.size();
    torch::Tensor features_tensor = torch::empty({num_rows, num_features_csv}, data_options);
    auto features_accessor = features_tensor.accessor<float, 2>();
    for (size_t i = 0; i < feature_rows.size(); i++) {
        for (int j = 0; j < num_features_csv; j++) {
            features_accessor[i][j] = feature_rows[i][j];
        }
    }
    torch::Tensor labels_tensor = torch::from_blob(labels.data(), {num_rows, 1}, data_options).clone();

    // ------------------------------
    // 3. Split data into train/val/test sets.
    // ------------------------------
    int64_t train_size = static_cast<int64_t>(train_ratio * num_rows);
    int64_t val_size = static_cast<int64_t>(val_ratio * num_rows);
    int64_t test_size = num_rows - train_size - val_size;
    auto train_features = features_tensor.slice(0, 0, train_size);
    auto train_labels = labels_tensor.slice(0, 0, train_size);
    auto val_features = features_tensor.slice(0, train_size, train_size + val_size);
    auto val_labels = labels_tensor.slice(0, train_size, train_size + val_size);
    auto test_features = features_tensor.slice(0, train_size + val_size, num_rows);
    auto test_labels = labels_tensor.slice(0, train_size + val_size, num_rows);

    // ------------------------------
    // 4. Partition features for left/right models.
    // ------------------------------
    auto [train_features_left, train_features_right] = verticalSplit(train_features, split_col);
    auto [val_features_left, val_features_right] = verticalSplit(val_features, split_col);
    auto [test_features_left, test_features_right] = verticalSplit(test_features, split_col);

    // ------------------------------
    // 5. Create datasets and dataloaders.
    // ------------------------------
    auto train_dataset = MyDataset(train_features, train_labels).map(torch::data::transforms::Stack<>());
    auto val_dataset = MyDataset(val_features, val_labels).map(torch::data::transforms::Stack<>());
    auto test_dataset = MyDataset(test_features, test_labels).map(torch::data::transforms::Stack<>());
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(batch_size)
    );
    auto val_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(val_dataset), torch::data::DataLoaderOptions().batch_size(batch_size)
    );
    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(batch_size)
    );

    // ------------------------------
    // 6. Initialize models and aggregator.
    // ------------------------------
    LRLocal lrLocalModel(left_input_dim, left_output_dim);
    MLP4Local mlpLocalModel(right_input_dim, right_output_dim);
    VFLAggregator vflAggregator(aggregator_input_dim, aggregator_output_dim);
    lrLocalModel->train();
    mlpLocalModel->train();
    vflAggregator->train();

    // ------------------------------
    // 7. Setup optimizer and loss.
    // ------------------------------
    std::vector<torch::optim::OptimizerParamGroup> param_groups;
    param_groups.push_back(lrLocalModel->parameters());
    param_groups.push_back(mlpLocalModel->parameters());
    param_groups.push_back(vflAggregator->parameters());
    torch::optim::Adam optimizer(param_groups, torch::optim::AdamOptions(learning_rate));
    auto criterion = torch::nn::BCEWithLogitsLoss();

    // Vectors to record epoch losses.
    std::vector<double> epoch_train_losses;
    std::vector<double> epoch_val_losses;
    double best_val_accuracy = 0.0;

    // ------------------------------
    // 8. Training loop with checkpointing.
    // ------------------------------
    std::cout << "Training for " << num_epochs << " epochs..." << std::endl;
    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        lrLocalModel->train();
        mlpLocalModel->train();
        vflAggregator->train();
        double running_loss = 0.0;
        size_t total_samples = 0;
        for (auto &batch : *train_loader) {
            auto inputs = batch.data;
            auto targets = batch.target;
            size_t bsize = inputs.size(0);
            total_samples += bsize;
            auto inputs_left = inputs.slice(/*dim=*/1, 0, split_col);
            auto inputs_right = inputs.slice(/*dim=*/1, split_col, inputs.size(1));
            auto local_out_left = lrLocalModel->forward(inputs_left);
            auto local_out_right = mlpLocalModel->forward(inputs_right);
            auto final_out = vflAggregator->forward({local_out_left, local_out_right});
            optimizer.zero_grad();
            auto loss = criterion(final_out, targets);
            loss.backward();
            optimizer.step();
            running_loss += loss.template item<double>() * bsize;
        }
        double epoch_loss = running_loss / total_samples;
        epoch_train_losses.push_back(epoch_loss);
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                  << " - Training Loss: " << epoch_loss << std::endl;

        // Validation loop.
        lrLocalModel->eval();
        mlpLocalModel->eval();
        vflAggregator->eval();
        double val_loss_sum = 0.0;
        size_t val_samples = 0;
        for (auto &batch : *val_loader) {
            auto inputs = batch.data;
            auto targets = batch.target;
            size_t bsize = inputs.size(0);
            val_samples += bsize;
            auto inputs_left = inputs.slice(/*dim=*/1, 0, split_col);
            auto inputs_right = inputs.slice(/*dim=*/1, split_col, inputs.size(1));
            auto local_out_left = lrLocalModel->forward(inputs_left);
            auto local_out_right = mlpLocalModel->forward(inputs_right);
            auto final_out = vflAggregator->forward({local_out_left, local_out_right});
            auto loss = criterion(final_out, targets);
            val_loss_sum += loss.template item<double>() * bsize;
        }
        double epoch_val_loss = val_loss_sum / val_samples;
        epoch_val_losses.push_back(epoch_val_loss);
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                  << " - Validation Loss: " << epoch_val_loss << std::endl;

        // Compute validation accuracy.
        auto val_loader_seq = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            MyDataset(val_features, val_labels).map(torch::data::transforms::Stack<>()),
            torch::data::DataLoaderOptions().batch_size(batch_size)
        );
        double current_val_accuracy = compute_accuracy(lrLocalModel, mlpLocalModel, vflAggregator, *val_loader_seq, split_col);
        std::cout << "Epoch " << (epoch + 1) << " - Validation Accuracy: " << current_val_accuracy * 100 << "%" << std::endl;
        if (current_val_accuracy > best_val_accuracy) {
            best_val_accuracy = current_val_accuracy;
            torch::save(lrLocalModel, run_folder_path + "/best_lrLocalModel.pt");
            torch::save(mlpLocalModel, run_folder_path + "/best_mlpLocalModel.pt");
            torch::save(vflAggregator, run_folder_path + "/best_vflAggregator.pt");
            std::cout << "Checkpoint: Best model updated at epoch " << (epoch + 1) << std::endl;
        }
    }

    // ------------------------------
    // 9. Load best model for testing.
    // ------------------------------
    std::cout << "\nLoading best model for testing (Validation Accuracy: " << best_val_accuracy * 100 << "%)..." << std::endl;
    torch::load(lrLocalModel, run_folder_path + "/best_lrLocalModel.pt");
    torch::load(mlpLocalModel, run_folder_path + "/best_mlpLocalModel.pt");
    torch::load(vflAggregator, run_folder_path + "/best_vflAggregator.pt");

    // ------------------------------
    // 10. Testing loop.
    // ------------------------------
    lrLocalModel->eval();
    mlpLocalModel->eval();
    vflAggregator->eval();
    double test_loss_sum = 0.0;
    size_t test_samples = 0;
    for (auto &batch : *test_loader) {
        auto inputs = batch.data;
        auto targets = batch.target;
        size_t bsize = inputs.size(0);
        test_samples += bsize;
        auto inputs_left = inputs.slice(/*dim=*/1, 0, split_col);
        auto inputs_right = inputs.slice(/*dim=*/1, split_col, inputs.size(1));
        auto local_out_left = lrLocalModel->forward(inputs_left);
        auto local_out_right = mlpLocalModel->forward(inputs_right);
        auto final_out = vflAggregator->forward({local_out_left, local_out_right});
        auto loss = criterion(final_out, targets);
        test_loss_sum += loss.template item<double>() * bsize;
    }
    double test_loss = test_loss_sum / test_samples;
    std::cout << "Test Loss: " << test_loss << std::endl;

    // ------------------------------
    // 11. Compute accuracies.
    // ------------------------------
    auto train_loader_seq = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        MyDataset(train_features, train_labels).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );
    auto val_loader_seq = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        MyDataset(val_features, val_labels).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );
    auto test_loader_seq = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        MyDataset(test_features, test_labels).map(torch::data::transforms::Stack<>()),
        torch::data::DataLoaderOptions().batch_size(batch_size)
    );

    double train_accuracy = compute_accuracy(lrLocalModel, mlpLocalModel, vflAggregator, *train_loader_seq, split_col);
    double val_accuracy = compute_accuracy(lrLocalModel, mlpLocalModel, vflAggregator, *val_loader_seq, split_col);
    double test_accuracy = compute_accuracy(lrLocalModel, mlpLocalModel, vflAggregator, *test_loader_seq, split_col);

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
        log_file << "csv_path: " << csv_path << "\n";
        log_file << "num_csv_columns: " << num_csv_columns << "\n";
        log_file << "feature_start: " << feature_start << "\n";
        log_file << "num_features_csv: " << num_features_csv << "\n";
        log_file << "target_index: " << target_index << "\n";
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
