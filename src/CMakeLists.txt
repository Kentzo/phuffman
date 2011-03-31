cmake_minimum_required(VERSION 2.6)
project(PHUFFMAN)

find_package(CUDA)
include_directories(./CUDA/)
set(CUDA_NVCC_FLAGS "-arch=sm_13")
cuda_add_library(phuffman STATIC phuffman.cpp DepthCounterNode.cpp Code.c Frequency.c CodesTableAdapter.cpp ./CUDA/encode.cu ./CUDA/decode.cu)