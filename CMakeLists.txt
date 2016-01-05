# Set minimum cmake version
cmake_minimum_required(VERSION 2.6)

# Set compiler and flags
set(CMAKE_C_COMPILER "clang")
set(CMAKE_C_FLAGS "-Wall -O2")

# Enable test cases
enable_testing()

# Setup project
project(vecotr_ann C)

# Add include dir
include_directories(include)

# Add subdirectories
add_subdirectory(lib)
add_subdirectory(cmake)
add_subdirectory(tests)
