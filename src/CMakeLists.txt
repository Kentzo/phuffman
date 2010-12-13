if (APPLE)
  find_library(OPENCL_LIBRARY OpenCL DOC "OpenCL lib for OSX")
  find_path(OPENCL_INCLUDE_DIR OpenCL/cl.h DOC "Include for OpenCL on OSX")
endif (APPLE)

add_library(phuffman STATIC phuffman.cpp DepthCounterNode.cpp Code.c Frequency.c CodeTableAdapter.cpp verbose.cpp)
target_link_libraries(phuffman ${OPENCL_LIBRARY})