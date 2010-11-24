include_directories(${PHUFFMAN_SOURCE_DIR}/src)
link_directories(${PHUFFMAN_SOURCE_DIR}/src)
add_executable(demo main.cpp)
target_link_libraries(demo phuffman)
