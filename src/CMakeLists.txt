cmake_minimum_required(VERSION 2.6)

include_directories(./3rdparty/ ./)

add_library (phuffman STATIC phuffman.cpp DepthCounterNode.cpp Code.c Frequency.c CodesTableAdapter.cpp)
add_subdirectory(CUDA)
link_directories(./CUDA)
target_link_libraries(phuffman cuda_encode)
