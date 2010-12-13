cmake_minimum_required(VERSION 2.8)
project(PHUFFMAN)

add_subdirectory(src)

set(SRCS ${PHUFFMAN_SOURCE_DIR}/demo/main.cpp)

if (OPENCL_ENABLED)
  if (UNIX)
    if (APPLE)
      find_library(OPENCL_LIBRARY OpenCL DOC "OpenCL lib for OSX")
      find_path(OPENCL_INCLUDE_DIR OpenCL/cl.h DOC "Include for OpenCL on OSX")
    else (APPLE)
      find_library(OPENCL_LIBRARY OpenCL ENV LD_LIBRARY_PATH DOC "NVIDIA OpenCL lib for Unix-like")
      find_path(OPENCL_INCLUDE_DIR CL/opencl.h $ENV{HOME}/NVIDIA_GPU_Computing_SDK/OpenCL/common/inc DOC "Include for NVIDIA OpenCL on Unix-like")
    endif (APPLE)
  elseif (UNIX)
    if (WIN32)
    elseif (WIN32)
    endif(WIN32)
  endif (UNIX)
  include_directories(${OPENCL_INCLUDE_DIR})
  include_directories(${PHUFFMAN_SOURCE_DIR}/src/OpenCL)
endif (OPENCL_ENABLED)

if (CUDA_ENABLED)
  include_directories(${PHUFFMAN_SOURCE_DIR}/src/CUDA)
endif (CUDA_ENABLED)

include_directories(${PHUFFMAN_SOURCE_DIR}/src)
add_executable(demo ${PHUFFMAN_SOURCE_DIR}/demo/main.cpp)

link_directories(${PHUFFMAN_SOURCE_DIR}/src)
target_link_libraries(demo phuffman)
