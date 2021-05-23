cmake_minimum_required(VERSION 3.19)
project(libtosa)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(libtosa)
add_subdirectory(test)