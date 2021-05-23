cmake_minimum_required(VERSION 3.19)
project(cli_test)

set(CMAKE_CXX_STANDARD 14)

include_directories(${CMAKE_SOURCE_DIR}/libtosa/include)
add_executable(cli_test main.cpp)
target_link_libraries(cli_test PUBLIC tosa)