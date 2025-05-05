cmake_minimum_required(VERSION 3.8)
project(LightweightCUDAMLP CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES "75-real;86")

# Enable relevant compilation flags
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

add_executable(lightweight_cudamlp
    main.cu
)

# We include the current directory so that #include "XYZ.cuh" will work
target_include_directories(lightweight_cudamlp PUBLIC
    ${PROJECT_SOURCE_DIR}
)