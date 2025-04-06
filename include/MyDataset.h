#ifndef MYDATASET_H
#define MYDATASET_H

#include <Eigen/Dense>
#include <vector>
#include <random>
#include <algorithm>
#include <iostream>

// Structure to hold a single example
struct EigenExample {
    Eigen::MatrixXd data;    // Features (rows: feature dim, cols: 1)
    Eigen::MatrixXd target;  // Target (rows: target dim, cols: 1)
};

// Structure to hold a batch of examples
struct EigenBatch {
    Eigen::MatrixXd data;    // Features (rows: feature dim, cols: batch_size)
    Eigen::MatrixXd target;  // Target (rows: target dim, cols: batch_size)
};

// Dataset class for Eigen matrices
class EigenDataset {
private:
    Eigen::MatrixXd data;    // Features (rows: feature dim, cols: num_samples)
    Eigen::MatrixXd target;  // Target (rows: target dim, cols: num_samples)
    
public:
    EigenDataset(const Eigen::MatrixXd& data, const Eigen::MatrixXd& target)
        : data(data), target(target) {}
    
    EigenExample get(size_t idx) const {
        if (idx >= data.cols()) {
            std::cerr << "Index out of bounds: " << idx << " >= " << data.cols() << std::endl;
            return {Eigen::MatrixXd::Zero(data.rows(), 1), Eigen::MatrixXd::Zero(target.rows(), 1)};
        }
        return {data.col(idx), target.col(idx)};
    }
    
    size_t size() const {
        return data.cols();
    }
    
    size_t data_dim() const {
        return data.rows();
    }
    
    size_t target_dim() const {
        return target.rows();
    }
    
    // Getter methods for direct access
    const Eigen::MatrixXd& getData() const { return data; }
    const Eigen::MatrixXd& getTarget() const { return target; }
};

// DataLoader class for batched processing
class EigenDataLoader {
private:
    const EigenDataset& dataset;
    size_t batch_size;
    bool shuffle_data;
    std::vector<size_t> indices;
    
public:
    class Iterator {
    private:
        const EigenDataLoader& data_loader;
        size_t current_index;
        std::vector<size_t> indices;
        
    public:
        Iterator(const EigenDataLoader& loader, size_t current, const std::vector<size_t>& indices)
            : data_loader(loader), current_index(current), indices(indices) {}
        
        EigenBatch operator*() const {
            try {
                // Calculate real batch size (can be smaller for the last batch)
                size_t remaining = data_loader.dataset.size() - current_index;
                size_t actual_batch_size = std::min(data_loader.batch_size, remaining);
                
                // Return empty batch if we're at the end
                if (actual_batch_size == 0 || current_index >= indices.size()) {
                    return {Eigen::MatrixXd::Zero(0, 0), Eigen::MatrixXd::Zero(0, 0)};
                }
                
                // Create matrices for batch (always use full batch size for consistent dimensions)
                Eigen::MatrixXd batch_data = Eigen::MatrixXd::Zero(data_loader.dataset.data_dim(), data_loader.batch_size);
                Eigen::MatrixXd batch_target = Eigen::MatrixXd::Zero(data_loader.dataset.target_dim(), data_loader.batch_size);
                
                // Fill with actual data
                for (size_t i = 0; i < actual_batch_size && current_index + i < indices.size(); ++i) {
                    size_t idx = indices[current_index + i];
                    if (idx < data_loader.dataset.size()) {
                        batch_data.col(i) = data_loader.dataset.getData().col(idx);
                        batch_target.col(i) = data_loader.dataset.getTarget().col(idx);
                    } else {
                        std::cerr << "Warning: Dataset index out of bounds: " << idx << " >= " << data_loader.dataset.size() << std::endl;
                        // Leave as zeros (already initialized that way)
                    }
                }
                
                // If we used less than the full batch size, we've implicitly padded with zeros
                if (actual_batch_size < data_loader.batch_size) {
                    std::cout << "Padding batch: " << actual_batch_size << " -> " << data_loader.batch_size << std::endl;
                }
                
                return {batch_data, batch_target};
            } catch (const std::exception& e) {
                std::cerr << "Exception in Iterator::operator*: " << e.what() << std::endl;
                return {Eigen::MatrixXd::Zero(0, 0), Eigen::MatrixXd::Zero(0, 0)};
            }
        }
        
        Iterator& operator++() {
            current_index += data_loader.batch_size;
            return *this;
        }
        
        bool operator!=(const Iterator& other) const {
            return current_index != other.current_index;
        }
    };
    
    EigenDataLoader(const EigenDataset& dataset, size_t batch_size, bool shuffle = false)
        : dataset(dataset), batch_size(batch_size), shuffle_data(shuffle) {
            
        // Initialize indices
        indices.resize(dataset.size());
        for (size_t i = 0; i < dataset.size(); ++i) {
            indices[i] = i;
        }
        
        if (shuffle_data) {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::shuffle(indices.begin(), indices.end(), gen);
        }
    }
    
    void shuffle() {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices.begin(), indices.end(), g);
    }
    
    size_t num_batches() const {
        return (dataset.size() + batch_size - 1) / batch_size;
    }
    
    size_t size() const {
        return dataset.size();
    }
    
    Iterator begin() {
        if (shuffle_data) {
            shuffle();
        }
        return Iterator(*this, 0, indices);
    }
    
    Iterator end() {
        // Change to use ceil(dataset.size() / batch_size) * batch_size as the end index
        size_t num_complete_batches = (dataset.size() + batch_size - 1) / batch_size;
        return Iterator(*this, num_complete_batches * batch_size, indices);
    }
};

#endif // MYDATASET_H

