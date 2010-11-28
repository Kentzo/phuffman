cmake_minimum_required(VERSION 2.6)
project(PHUFFMAN)

set(CMAKE_CXX_FLAGS "-ggdb -Wall")
set(CMAKE_BINARY_DIR ${PHUFFMAN_SOURCE_DIR}/build)

include_directories(${PHUFFMAN_SOURCE_DIR}/src/POD)

add_subdirectory(src)
add_subdirectory(demo)
