cmake_minimum_required(VERSION 2.6)
project(PHUFFMAN)

add_subdirectory(src)

add_custom_command(
  OUTPUT OpenCL
  COMMAND ${CMAKE_COMMAND} -E copy ${PHUFFMAN_SOURCE_DIR}/src/OpenCL/encoding.cl ${CMAKE_CURRENT_BINARY_DIR}/encoding.cl
  COMMAND ${CMAKE_COMMAND} -E copy ${PHUFFMAN_SOURCE_DIR}/src/Code.h ${CMAKE_CURRENT_BINARY_DIR}/Code.h
)

include_directories(${PHUFFMAN_SOURCE_DIR}/src)
link_directories(${PHUFFMAN_SOURCE_DIR}/src)
add_executable(demo ${PHUFFMAN_SOURCE_DIR}/demo/main.cpp OpenCL)
target_link_libraries(demo phuffman)