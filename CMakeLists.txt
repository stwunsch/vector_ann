# Set minimum cmake version
cmake_minimum_required(VERSION 2.6)

# Setup project
project(vecotr_ann C)

# Set compiler and flags
set(CMAKE_C_COMPILER "gcc")
set(CMAKE_C_FLAGS "-Wall -O2 -fPIC")

# Enable test cases
enable_testing()

# Add include dir
include_directories(include)

# Add subdirectories
add_subdirectory(lib)
add_subdirectory(cmake)
add_subdirectory(tests)
