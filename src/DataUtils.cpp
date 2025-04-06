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
    
    std::cout << "Loading CSV file: " << filename << std::endl;
    
    // Check if this is a neurips dataset
    bool is_neurips = filename.find("neurips") != std::string::npos;
    
    std::string line;
    bool skip_header = true;
    size_t line_count = 0;
    size_t expected_columns = 0;
    
    while (std::getline(file, line)) {
        line_count++;
        
        // Skip empty lines
        if (line.empty()) {
            std::cout << "Warning: Skipping empty line at line " << line_count << std::endl;
            continue;
        }
        
        // Process header line
        if (skip_header) {
            skip_header = false;
            
            if (is_neurips) {
                // For neurips, the header is long but we know it has 32 columns
                expected_columns = 32;
                std::cout << "NeurIPS dataset detected. Using 32 columns." << std::endl;
            } else {
                auto header_tokens = split(line, ',');
                expected_columns = header_tokens.size();
                std::cout << "CSV header detected with " << expected_columns << " columns" << std::endl;
            }
            continue;
        }
        
        auto tokens = split(line, ',');
        
        // For neurips data, enforce the expected number of columns
        if (is_neurips && tokens.size() != expected_columns) {
            std::cerr << "Warning: Line " << line_count << " has " << tokens.size() 
                      << " columns, expected " << expected_columns << ". Skipping." << std::endl;
            continue;
        }
        
        // For non-neurips data, provide a warning but still process
        if (!is_neurips && tokens.size() != expected_columns) {
            std::cout << "Warning: Line " << line_count << " has " << tokens.size() 
                      << " columns, expected " << expected_columns << ". Attempting to process anyway." << std::endl;
        }
        
        data.push_back(tokens);
    }
    
    file.close();
    std::cout << "Loaded " << data.size() << " data rows from CSV" << std::endl;
    return data;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> verticalSplit(const Eigen::MatrixXd &full_features, int64_t split_col) {
    std::cout << "verticalSplit input dimensions: " << full_features.rows() << "x" << full_features.cols() << std::endl;
    std::cout << "Split column: " << split_col << std::endl;
    
    Eigen::MatrixXd left = full_features.topRows(split_col);
    Eigen::MatrixXd right = full_features.bottomRows(full_features.rows() - split_col);
    
    std::cout << "verticalSplit output dimensions - Left: " << left.rows() << "x" << left.cols() 
              << ", Right: " << right.rows() << "x" << right.cols() << std::endl;
    
    return {left, right};
}

void shuffleData(std::vector<std::vector<float>> &features, std::vector<float> &labels) {
    std::random_device rd;
    std::mt19937 g(rd());
    
    // Create a vector of indices
    std::vector<size_t> indices(features.size());
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = i;
    }
    
    // Shuffle the indices
    std::shuffle(indices.begin(), indices.end(), g);
    
    // Create temporary vectors for the shuffled data
    std::vector<std::vector<float>> shuffled_features(features.size());
    std::vector<float> shuffled_labels(labels.size());
    
    // Populate the shuffled vectors
    for (size_t i = 0; i < indices.size(); i++) {
        shuffled_features[i] = features[indices[i]];
        shuffled_labels[i] = labels[indices[i]];
    }
    
    // Copy back to the original vectors
    features = shuffled_features;
    labels = shuffled_labels;
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> convertToEigenMatrices(
    const std::vector<std::vector<float>> &features, 
    const std::vector<float> &labels) {
    
    // Check for empty data
    if (features.empty() || labels.empty()) {
        std::cerr << "ERROR: Empty features or labels vector" << std::endl;
        return {Eigen::MatrixXd(0, 0), Eigen::MatrixXd(0, 0)};
    }
    
    // Determine dimensions
    int64_t num_features = features[0].size();
    int64_t num_samples = features.size();
    
    // Initialize Eigen matrices
    Eigen::MatrixXd features_matrix(num_features, num_samples);
    Eigen::MatrixXd labels_matrix(1, num_samples);
    
    // Fill the matrices
    for (int64_t i = 0; i < num_samples; i++) {
        if (features[i].size() != static_cast<size_t>(num_features)) {
            std::cerr << "ERROR: Inconsistent feature vector size at index " << i 
                      << ". Expected " << num_features << ", got " << features[i].size() << std::endl;
            continue;
        }
        
        for (int64_t j = 0; j < num_features; j++) {
            features_matrix(j, i) = features[i][j];
        }
        
        labels_matrix(0, i) = labels[i];
    }
    
    return {features_matrix, labels_matrix};
}
