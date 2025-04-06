# 🚀 Vertical Federated Learning (VFL) in C++ 🔒

## 📖 Project Overview

This repository provides a comprehensive C++ implementation of **Vertical Federated Learning (VFL)**, a privacy-preserving technique allowing multiple parties to collaboratively train machine learning models without exposing raw data. Designed for binary classification tasks, this implementation demonstrates secure vertical data partitioning and efficient model training across distributed datasets.

🎯 **Objectives:**

- 🔒 **Privacy Preservation:** Enable secure collaboration without sharing sensitive data.
- 📈 **Model Performance:** Match or surpass centralized training benchmarks.
- 🛠️ **Flexibility:** Support diverse model architectures and datasets.
- ⚡ **Efficiency:** Optimize matrix operations and training performance.
- 📚 **Reproducibility:** Offer a well-documented and easy-to-follow codebase.

## 📂 Codebase Structure

```
vertical-federated-learning-cpp/
├── 📄 CMakeLists.txt           # Build configuration
├── 📖 README.md                # Documentation
├── 📁 data/                    # Datasets
│   ├── 📁 credit/              # Credit card datasets
│   └── 📁 neurips/             # NeurIPS fraud datasets
├── 📁 include/                 # Header files
│   ├── DataUtils.h             # Data processing utilities
│   ├── EvaluateUtils.h         # Evaluation metrics
│   ├── LocalModels.h           # Logistic Regression and MLP implementations
│   ├── MyDataset.h             # Dataset handling
│   ├── TrainUtils.h            # Training utilities
│   ├── VFLBaseLocalModel.h     # Base class for local models
│   └── VFLAggregator.h         # Combines local outputs
├── 📁 libtorch/                # LibTorch library
├── 📁 python/                  # Visualization scripts
│   └── plot_losses.py          # Plots training/validation loss curves
├── 📁 runs/                    # Run results
├── 📁 src/                     # Source files
│   ├── DataUtils.cpp           # Data utilities implementation
│   └── main.cpp                # Entry point coordinating the VFL pipeline
└── 📄 eigen-example.cpp        # Eigen usage example
```

### 🔑 Key Components

#### 📄 Header Files

- **VFLBaseLocalModel.h**: Abstract base class for local models, defining the interface for forward and backward passes.
- **LocalModels.h**: Contains implementations of local models (Logistic Regression and MLP) and activation functions.
- **VFLAggregator.h**: Implements the aggregator that combines outputs from local models.
- **MyDataset.h**: Defines dataset and dataloader classes for efficient data handling.
- **DataUtils.h**: Declares utilities for data loading, preprocessing, and splitting.
- **EvaluateUtils.h**: Contains functions for computing accuracy and other evaluation metrics.
- **TrainUtils.h**: Provides utilities for training and model parameter management.

#### 📄 Source Files

- **main.cpp**: The main program that orchestrates the entire VFL process, from data loading to model training and evaluation.
- **DataUtils.cpp**: Implementation of data utilities declared in DataUtils.h.

#### 🐍 Python Scripts

- **plot_losses.py**: Script for visualizing training and validation losses over epochs.

## 🧠 Model Architecture

The VFL implementation uses a split neural network architecture where:

1. **Data Partitioning**: Features are vertically split between parties (e.g., left half and right half of features).
2. **Local Models**: Each party trains a local model on their portion of features.
3. **Aggregation**: A central aggregator combines the outputs of local models to produce the final prediction.

### 🔍 Detailed Architecture

#### 🧮 Local Models

1. **Logistic Regression (LRLocal)**:
   - Input dimension: `left_input_dim` (half of total features)
   - Output dimension: `left_output_dim` (64 by default)
   - Architecture: Linear transformation (W*x + b)
   - No activation function (activation is applied at the aggregator level)

2. **Multi-Layer Perceptron (MLP4Local)**:
   - Input dimension: `right_input_dim` (remaining half of features)
   - Hidden dimension: 100 (configurable)
   - Output dimension: `right_output_dim` (64 by default)
   - Architecture: 4-layer MLP with ReLU activation between layers
   - No activation function on the output layer

#### 🔄 Aggregator (VFLAggregator)

- Input dimension: `aggregator_input_dim` (sum of local output dimensions, 128 by default)
- Output dimension: `aggregator_output_dim` (1 for binary classification)
- Architecture: Linear transformation (W*concatenated + b)
- Activation: Sigmoid (applied after the linear transformation)

### 🔄 Data Flow

1. Input data is vertically split into left and right portions.
2. Left portion is fed into the LRLocal model.
3. Right portion is fed into the MLP4Local model.
4. Outputs from both local models are concatenated.
5. Concatenated output is fed into the VFLAggregator.
6. Final output is passed through a sigmoid activation for binary classification.

## 🔧 Implementation Details

### 📊 Data Handling

The implementation uses Eigen matrices for efficient numerical computations. Data is loaded from CSV files and converted to Eigen matrices for processing. The `EigenDataset` and `EigenDataLoader` classes provide a PyTorch-like interface for data handling, supporting batching and shuffling.

### 🧮 Model Implementation

Local models and the aggregator are implemented as C++ classes with forward and backward methods. The forward method computes the output given an input, while the backward method computes gradients and updates parameters.

#### ➡️ Forward Pass

1. Input data is split vertically.
2. Each local model processes its portion of the data.
3. Outputs are concatenated and fed into the aggregator.
4. Final output is computed.

#### ⬅️ Backward Pass

1. Gradient is computed at the output layer.
2. Gradient is propagated back through the aggregator.
3. Aggregator splits the gradient and distributes it to local models.
4. Local models update their parameters using the received gradients.

### 📉 Loss Function

The implementation uses Binary Cross Entropy (BCE) loss for binary classification:

```
BCE = -y * log(p) - (1-y) * log(1-p)
```

where `y` is the true label and `p` is the predicted probability.

### 🔄 Optimization

Stochastic Gradient Descent (SGD) is used for optimization, with a configurable learning rate. The implementation supports mini-batch training for improved efficiency.

### 💾 Model Serialization

Model parameters are serialized using LibTorch for compatibility with PyTorch models. This allows for easy integration with PyTorch-based systems.

## ⚙️ Environment Setup

### 📋 Prerequisites
- 🛠️ **C++17 compiler** (GCC/Clang/MSVC)
- 🛠️ **CMake ≥ 3.10**
- 📐 **Eigen ≥ 3.3**
- 🧩 **LibTorch** (included)
- 🐍 **Python ≥ 3.6** (for plotting)
- 🚀 **OpenMP** (parallelization)

### 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vertical-federated-learning-cpp.git
   cd vertical-federated-learning-cpp
   ```

2. Build the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   make -j4
   ```

3. Install Python dependencies (for plotting):
   ```bash
   pip install pandas matplotlib
   ```

### 🔧 Environment Variables

On macOS, you may need to set the `DYLD_LIBRARY_PATH` to include the LibTorch library path:
```bash
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH
```

## 📊 Dataset and Preprocessing

### 💳 Credit Card Default Dataset
- **Source:** UCI Machine Learning Repository
- **Description:** Predicts credit card default based on payment history
- **Features:** 23 numerical features
- **Target:** Binary (default or not)
- **Preprocessing:** Features are normalized, and the dataset is split into train/validation/test sets

### 🧐 NeurIPS Fraud Detection Dataset
- **Source:** NeurIPS 2022 Competition
- **Description:** Detects fraudulent transactions
- **Features:** 31 numerical features
- **Target:** Binary (fraudulent or not)
- **Preprocessing:** Features are normalized, and the dataset is split into train/validation/test sets

### 🔄 Data Preprocessing
1. **Loading:** CSV files loaded via `loadCSV`.
2. **Parsing:** CSV data converted to features/labels.
3. **Shuffling:** Ensures data randomness.
4. **Splitting:** Train/Val/Test (70%/15%/15%).
5. **Vertical Partitioning:** Features split vertically.

## ▶️ Running the Codebase

### 🚀 Basic Usage

Run the program with the default dataset (credit):
```bash
./vfl_main
```

### 🔧 Command-line Options

- `--data`: Specify the dataset to use (default: "credit")
  - Available options: "credit", "credit-balanced", "neurips-base", "neurips-v1", "neurips-v2", "neurips-v3", "neurips-v4", "neurips-v5"

Example:
```bash
./vfl_main --data neurips-base
```

### 📊 Output

The program creates a timestamped directory in the `runs` folder containing:
- Best model parameters (saved as PyTorch tensors)
- Loss curves (CSV file and plots)
- Run details (hyperparameters and final metrics)

## 📈 Training and Evaluation

### 🔄 Training Process

1. **Initialization**: Models are initialized with random weights.
2. **Epoch Loop**: For each epoch:
   - **Training**: Process batches of training data, compute loss, and update parameters.
   - **Validation**: Evaluate the model on validation data and compute accuracy.
   - **Checkpointing**: Save the model if validation accuracy improves.
3. **Testing**: Evaluate the best model on test data.

### ⚙️ Hyperparameters

- **Batch Size**: 128 (configurable)
- **Learning Rate**: 0.001 (configurable)
- **Number of Epochs**: 10 (configurable)
- **Train/Val/Test Split**: 70%/15%/15% (configurable)
- **Local Output Dimensions**: 64 (configurable)
- **Aggregator Output Dimension**: 1 (for binary classification)

### 📊 Evaluation Metrics

- **Loss**: Binary Cross Entropy (BCE)
- **Accuracy**: Percentage of correctly classified samples

## 📊 Performance Analysis

The implementation includes visualization tools for analyzing model performance:

### 📈 Loss Curves

Training and validation losses are plotted over epochs to visualize the learning process. The plots are saved as JPEG images in the run directory.

### 📊 Accuracy Metrics

Final accuracy metrics are reported for train, validation, and test sets, providing a comprehensive view of model performance.

## 🚢 Model Deployment

1. **Load trained models:**
```cpp
std::vector<torch::Tensor> lr_params, mlp_params, agg_params;
torch::load(lr_params, "path/to/best_lrLocalModel.pt");
torch::load(mlp_params, "path/to/best_mlpLocalModel.pt");
torch::load(agg_params, "path/to/best_vflAggregator.pt");

LRLocal lr_model(left_input_dim, left_output_dim);
MLP4Local mlp_model(right_input_dim, hidden_dim, right_output_dim);
VFLAggregator agg_model(local_output_dims, aggregator_output_dim);

lr_model.set_parameters(lr_params);
mlp_model.set_parameters(mlp_params);
agg_model.set_parameters(agg_params);
```

2. **Make predictions:**
```cpp
// Split input data
auto [inputs_left, inputs_right] = verticalSplit(input_data, split_col);

// Forward pass
auto local_out_left = lr_model.forward(inputs_left);
auto local_out_right = mlp_model.forward(inputs_right);
auto final_out = agg_model.forward({local_out_left, local_out_right});

// Apply sigmoid for probabilities
Eigen::MatrixXd probabilities = sigmoid(final_out);
```

## 📊 Dataset and Preprocessing

### Credit Card Default Dataset
- **Source:** UCI Machine Learning Repository
- **Description:** Predicts credit card default based on payment history
- **Features:** 23 numerical features
- **Target:** Binary (default or not)

### NeurIPS Fraud Detection Dataset
- **Source:** NeurIPS 2022 Competition
- **Description:** Detects fraudulent transactions
- **Features:** 31 numerical features
- **Target:** Binary (fraudulent or not)

### Data Preprocessing
1. **Loading:** CSV files loaded via `loadCSV`.
2. **Parsing:** CSV data converted to features/labels.
3. **Shuffling:** Ensures data randomness.
4. **Splitting:** Train/Val/Test (70%/15%/15%).
5. **Vertical Partitioning:** Features split vertically.

---
✨ Happy Federated Learning! 🚀

