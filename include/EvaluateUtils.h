#ifndef EVALUATEUTILS_H
#define EVALUATEUTILS_H

#include <torch/torch.h>
#include "LocalModels.h"
#include "VFLAggregator.h"
#include "MyDataset.h"

// Template function to compute binary accuracy over a DataLoader.
template <typename DataLoader>
double compute_accuracy(LRLocal &lrModel, MLP4Local &mlpModel, VFLAggregator &agg,
                        DataLoader &loader, int64_t split_col) {
    size_t total_correct = 0;
    size_t total = 0;
    for (auto &batch : loader) {
        auto inputs = batch.data;
        auto targets = batch.target;
        auto inputs_left = inputs.slice(/*dim=*/1, 0, split_col);
        auto inputs_right = inputs.slice(/*dim=*/1, split_col, inputs.size(1));
        auto local_out_left = lrModel->forward(inputs_left);
        auto local_out_right = mlpModel->forward(inputs_right);
        auto final_out = agg->forward({local_out_left, local_out_right});
        auto preds = torch::sigmoid(final_out) > 0.5;
        auto correct = preds.eq(targets > 0.5).sum().template item<int64_t>();
        total_correct += correct;
        total += targets.numel();
    }
    return static_cast<double>(total_correct) / total;
}

#endif // EVALUATEUTILS_H

