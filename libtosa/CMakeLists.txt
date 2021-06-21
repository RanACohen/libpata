cmake_minimum_required(VERSION 3.19)
project(libtosa)

set(CMAKE_CXX_STANDARD 14)
file(GLOB SRC_FILES src/*.c src/*.cpp)
file(GLOB SRC_BACKENDS_FILES src/backends/*.c src/backends/*.cpp src/backends/*.h)
file(GLOB SRC_CPU_BACKENDS_FILES src/backends/cpu/*.[ch]*)
file(GLOB INC_FILES include/*.h include/*.hpp)

include_directories(include)
include_directories(src/backends)
include_directories(src/utils)

add_library(tosa SHARED ${SRC_FILES} ${SRC_BACKENDS_FILES} ${SRC_CPU_BACKENDS_FILES} ${INC_FILES} include/tosa_errors.h)
