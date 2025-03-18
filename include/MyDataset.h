#ifndef MYDATASET_H
#define MYDATASET_H

#include <torch/torch.h>

// Custom dataset that provides data and target tensors.
struct MyDataset : torch::data::datasets::Dataset<MyDataset> {
    torch::Tensor data_, targets_;

    MyDataset(torch::Tensor data, torch::Tensor targets)
        : data_(std::move(data)), targets_(std::move(targets)) {}

    // Returns one sample (data and target) at the given index.
    torch::data::Example<> get(size_t index) override {
        return {data_[index], targets_[index]};
    }

    // Returns the total number of samples.
    torch::optional<size_t> size() const override {
        return data_.size(0);
    }
};

#endif // MYDATASET_H

