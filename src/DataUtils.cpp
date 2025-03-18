#include "DataUtils.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>

std::vector<std::string> split(const std::string &s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::vector<std::string>> loadCSV(const std::string &filename) {
    std::vector<std::vector<std::string>> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file: " << filename << std::endl;
        return data;
    }
    std::string line;
    bool skip_header = true;
    while (std::getline(file, line)) {
        if (skip_header) {
            skip_header = false;
            continue;
        }
        auto tokens = split(line, ',');
        data.push_back(tokens);
    }
    file.close();
    return data;
}

std::pair<torch::Tensor, torch::Tensor> verticalSplit(const torch::Tensor &full_features, int64_t split_col) {
    auto left = full_features.slice(/*dim=*/1, 0, split_col);
    auto right = full_features.slice(/*dim=*/1, split_col, full_features.size(1));
    return {left, right};
}

void shuffleData(std::vector<std::vector<float>> &feature_rows, std::vector<float> &labels) {
    if (feature_rows.size() != labels.size()) return;
    // Create index vector.
    std::vector<size_t> indices(feature_rows.size());
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    // Shuffle indices using current time as seed.
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(seed));
    
    std::vector<std::vector<float>> shuffled_features;
    std::vector<float> shuffled_labels;
    shuffled_features.reserve(feature_rows.size());
    shuffled_labels.reserve(labels.size());
    for (auto i : indices) {
        shuffled_features.push_back(feature_rows[i]);
        shuffled_labels.push_back(labels[i]);
    }
    feature_rows = std::move(shuffled_features);
    labels = std::move(shuffled_labels);
}
