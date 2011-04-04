cmake_minimum_required(VERSION 2.6.4)
project(PHUFFMAN)
find_package(CUDA)

set(CMAKE_CXX_FLAGS "-ggdb -Wall")
set(EXECUTABLE_OUTPUT_PATH ${PHUFFMAN_SOURCE_DIR}/build)
set(CUDA_NVCC_FLAGS "-arch=sm_13")

include_directories(include)
cuda_add_library(phuffman STATIC src/phuffman.cpp src/DepthCounterNode.cpp src/Code.c src/Frequency.c src/CodesTableAdapter.cpp src/CUDA/encode.cu src/CUDA/decode.cu)

# Tests
add_subdirectory(test/gtest)
include_directories(test/gtest/include)
include_directories(src/CUDA)
cuda_add_executable(phuffman_unittest test/phuffman_unittest.cu test/cpu_encode.cpp test/cpu_decode.cpp)
target_link_libraries(phuffman_unittest phuffman gtest)

enable_testing()
add_test(phuffman_unittest ${EXECUTABLE_OUTPUT_PATH}/phuffman_unittest)
