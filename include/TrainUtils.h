#ifndef TRAINUTILS_H
#define TRAINUTILS_H

#include <torch/torch.h>
#include <vector>
#include <string>
#include <filesystem>
#include "LocalModels.h"
#include "VFLAggregator.h"
#include "MyDataset.h"
#include "EvaluateUtils.h"

// Structure to hold training results.
struct TrainResult {
    std::vector<double> train_losses;
    std::vector<double> val_losses;
    double best_val_accuracy;
};

// Template training function which runs the training loop with checkpointing.
template <typename TrainLoader, typename ValLoader>
TrainResult trainModel(LRLocal &lrModel, MLP4Local &mlpModel, VFLAggregator &agg,
                       TrainLoader &train_loader, ValLoader &val_loader,
                       torch::optim::Optimizer &optimizer, torch::nn::Module &criterion,
                       int64_t split_col, size_t num_epochs, const std::string &run_folder_path, int batch_size,
                       torch::Tensor &val_features, torch::Tensor &val_labels) {
    TrainResult result;
    result.best_val_accuracy = 0.0;
    for (size_t epoch = 0; epoch < num_epochs; epoch++) {
        lrModel->train();
        mlpModel->train();
        agg->train();
        double running_loss = 0.0;
        size_t total_samples = 0;
        for (auto &batch : train_loader) {
            auto inputs = batch.data;
            auto targets = batch.target;
            size_t bsize = inputs.size(0);
            total_samples += bsize;
            auto inputs_left = inputs.slice(/*dim=*/1, 0, split_col);
            auto inputs_right = inputs.slice(/*dim=*/1, split_col, inputs.size(1));
            auto local_out_left = lrModel->forward(inputs_left);
            auto local_out_right = mlpModel->forward(inputs_right);
            auto final_out = agg->forward({local_out_left, local_out_right});
            optimizer.zero_grad();
            // Use operator() to compute loss.
            auto loss = criterion(final_out, targets);
            loss.backward();
            optimizer.step();
            running_loss += loss.template item<double>() * bsize;
        }
        double epoch_loss = running_loss / total_samples;
        result.train_losses.push_back(epoch_loss);
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                  << " - Training Loss: " << epoch_loss << std::endl;

        // Validation loop.
        lrModel->eval();
        mlpModel->eval();
        agg->eval();
        double val_loss_sum = 0.0;
        size_t val_samples = 0;
        for (auto &batch : val_loader) {
            auto inputs = batch.data;
            auto targets = batch.target;
            size_t bsize = inputs.size(0);
            val_samples += bsize;
            auto inputs_left = inputs.slice(/*dim=*/1, 0, split_col);
            auto inputs_right = inputs.slice(/*dim=*/1, split_col, inputs.size(1));
            auto local_out_left = lrModel->forward(inputs_left);
            auto local_out_right = mlpModel->forward(inputs_right);
            auto final_out = agg->forward({local_out_left, local_out_right});
            auto loss = criterion(final_out, targets);
            val_loss_sum += loss.template item<double>() * bsize;
        }
        double epoch_val_loss = val_loss_sum / val_samples;
        result.val_losses.push_back(epoch_val_loss);
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                  << " - Validation Loss: " << epoch_val_loss << std::endl;

        // Compute validation accuracy using a sequential DataLoader.
        auto val_dataset_seq = MyDataset(val_features, val_labels).map(torch::data::transforms::Stack<>());
        auto val_loader_seq = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
            std::move(val_dataset_seq), torch::data::DataLoaderOptions().batch_size(batch_size)
        );
        double current_val_accuracy = compute_accuracy(lrModel, mlpModel, agg, *val_loader_seq, split_col);
        std::cout << "Epoch " << (epoch + 1) << " - Validation Accuracy: " << current_val_accuracy * 100 << "%" << std::endl;

        if (current_val_accuracy > result.best_val_accuracy) {
            result.best_val_accuracy = current_val_accuracy;
            torch::save(lrModel, run_folder_path + "/best_lrLocalModel.pt");
            torch::save(mlpModel, run_folder_path + "/best_mlpLocalModel.pt");
            torch::save(agg, run_folder_path + "/best_vflAggregator.pt");
            std::cout << "Checkpoint: Best model updated at epoch " << (epoch + 1) << std::endl;
        }
    }
    return result;
}

#endif // TRAINUTILS_H

