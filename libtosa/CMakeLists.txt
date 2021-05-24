cmake_minimum_required(VERSION 3.19)
project(libtosa)

set(CMAKE_CXX_STANDARD 14)
file(GLOB SRC_FILES src/*.c src/*.cpp)
file(GLOB INC_FILES include/*.h include/*.hpp)

include_directories(include)

add_library(tosa SHARED ${SRC_FILES} ${INC_FILES} include/tosa_errors.h)
