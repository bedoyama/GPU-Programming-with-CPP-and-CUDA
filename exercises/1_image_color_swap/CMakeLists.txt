cmake_minimum_required(VERSION 3.10)
project(color_swap CUDA CXX)

# Enable C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# Suppress warnings from third-party stb_image library
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-unused-variable")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 550")

# Add executables
add_executable(color_swap color_swap.cu)
add_executable(color_swap_randomized color_swap_randomized.cu)

# Set output directory for the executables
set_target_properties(color_swap color_swap_randomized PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Note: This project uses stb_image.h and stb_image_write.h
# Make sure to download the full implementations from:
# https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
# https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
