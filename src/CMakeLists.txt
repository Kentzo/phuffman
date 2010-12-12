add_library (phuffman STATIC phuffman.cpp DepthCounterNode.cpp Code.c Frequency.c CodesTableAdapter.cpp)
add_subdirectory(CUDA)
link_directories(${PHUFFMAN_SOURCE_DIR}/src/CUDA)
target_link_libraries(phuffman cuda_encode)
