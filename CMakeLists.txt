cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(VFL_CPP_Project)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(Torch_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_executable(vfl_main 
    src/main.cpp
    src/DataUtils.cpp
)

target_link_libraries(vfl_main "${TORCH_LIBRARIES}")

