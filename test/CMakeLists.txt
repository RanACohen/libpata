cmake_minimum_required(VERSION 3.19)
project(cli_test)

set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

include_directories(${CMAKE_SOURCE_DIR}/libpata/include)

enable_testing()

add_executable(cli_test test_tensors.cpp test_parallel.cpp)
target_link_libraries(cli_test PUBLIC pata gtest_main)

include(GoogleTest)
gtest_discover_tests(cli_test)
