project(PHUFFMAN_TESTS)
cmake_minimum_required(VERSION 2.6.4)

find_package(CUDA)
add_subdirectory(gtest)
include_directories(./../)
include_directories(./../CUDA/)
include_directories(./gtest/include/)
cuda_add_executable(phuffman_unittest phuffman_unittest.cu)
target_link_libraries(phuffman_unittest gtest)

enable_testing()
add_test(phuffman_unittest ${EXECUTABLE_OUTPUT_PATH}/phuffman_unittest)

