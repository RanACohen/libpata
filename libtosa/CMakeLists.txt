cmake_minimum_required(VERSION 3.19)
project(libcpptosa)

set(CMAKE_CXX_STANDARD 14)

add_library(cpptosa SHARED src/libmain.cpp)