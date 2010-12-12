FIND_PACKAGE(CUDA)

include_directories(/home/dlyfar/NVIDIA_GPU_Computing_SDK/C/common/inc ${PHUFFMAN_SOURCE_DIR}/src/)


CUDA_ADD_LIBRARY(cuda_encode
  encode.cu
  STATIC
  )
