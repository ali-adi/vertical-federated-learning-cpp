cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(VFL_CPP_Project)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

# Find Eigen package (assuming it's installed)
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(vfl_main 
    src/main.cpp
    src/DataUtils.cpp
)

target_link_libraries(vfl_main "${TORCH_LIBRARIES}")

# Add a README to explain command-line usage
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/README.md 
"# Vertical Federated Learning C++ Implementation

## Usage
```bash
./vfl_main --data [dataset_name]
```

Available datasets:
- credit (default): Default credit card client dataset
- credit-balanced: Balanced version of the credit card dataset
- neurips-base: NeurIPS Base dataset
- neurips-v1: NeurIPS Variant I dataset
- neurips-v2: NeurIPS Variant II dataset
- neurips-v3: NeurIPS Variant III dataset
- neurips-v4: NeurIPS Variant IV dataset
- neurips-v5: NeurIPS Variant V dataset
")

