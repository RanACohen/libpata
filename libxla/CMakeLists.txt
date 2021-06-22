cmake_minimum_required(VERSION 3.19)
project(libxla)

set(CMAKE_CXX_STANDARD 14)
file(GLOB SRC_FILES src/*.c src/*.cpp)
file(GLOB SRC_BACKENDS_FILES src/backends/*.c src/backends/*.cpp src/backends/*.h)
file(GLOB SRC_CPU_BACKENDS_FILES src/backends/cpu/*.[ch]*)
file(GLOB INC_FILES include/*.h include/*.hpp)

include_directories(include)
include_directories(src/backends)
include_directories(src/utils)

include(ExternalProject)
find_program(MAKE_EXE NAMES make gmake nmake)
ExternalProject_Add(
        libxsmm
	#        GIT_REPOSITORY git@github.com:hfp/libxsmm.git
	GIT_REPOSITORY https://github.com/hfp/libxsmm.git
        BUILD_COMMAND ${MAKE_EXE} -j8
        UPDATE_COMMAND ""
#        SOURCE_DIR ${make_lib_SOURCE_DIR}
        BUILD_IN_SOURCE 1
        LOG_BUILD 1
        CONFIGURE_COMMAND ""
        INSTALL_COMMAND "${make_lib_install_commands}"
)

ExternalProject_Get_property(libxsmm BINARY_DIR SOURCE_DIR)
include_directories(${SOURCE_DIR}/include)
link_directories(${BINARY_DIR}/lib/)

add_library(xla SHARED ${SRC_FILES} ${SRC_BACKENDS_FILES} ${SRC_CPU_BACKENDS_FILES} ${INC_FILES})
add_dependencies(xla libxsmm)
target_link_libraries(xla ${BINARY_DIR}/lib/libxsmm.a ${BINARY_DIR}/lib/libxsmmext.a ${BINARY_DIR}/lib/libxsmmnoblas.a dl)
