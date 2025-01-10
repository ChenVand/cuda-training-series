cmake_minimum_required(VERSION 3.12)
project(helloCUDA)

# Find CUDA package
find_package(CUDA REQUIRED)

# Set the target executable
set(TARGET helloCUDA)

# Set the source files
set(SRC_FILES src/matrix_sums.cu)

# Add the executable target
cuda_add_executable(${TARGET} ${SRC_FILES})
