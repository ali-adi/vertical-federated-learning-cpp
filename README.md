# VFL_CPP_Project

This project is a simple simulation of vertical federated learning (VFL) implemented in C++ using LibTorch.

## Folder Structure

- **build/**: Directory where CMake will generate build files.
- **data/**: Directory to store your data files (e.g., CSV files).
- **include/**: Header files for the project.
- **libtorch/**: LibTorch library (download from [PyTorch C++ API](https://pytorch.org/cppdocs/installing.html)).
- **src/**: Source files (contains `main.cpp`).
- **CMakeLists.txt**: CMake configuration file.
- **README.md**: This file.

## Building the Project

1. **Install Prerequisites**:
   - **C++ Compiler**: Ensure you have a C++14 compiler installed (e.g., g++ or Visual Studio).
   - **CMake**: Version 3.10 or later.
   - **LibTorch**: Download and extract LibTorch to the `libtorch/` folder or adjust the path in `CMakeLists.txt`.

2. **Build Steps**:
   Open a terminal in the project root directory and run:

   ```bash
   mkdir build
   cd build
   cmake -DCMAKE_PREFIX_PATH=../libtorch ..
   make

   To run:
   ./vfl_main

