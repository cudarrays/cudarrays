project(cudarrays2)
cmake_minimum_required(VERSION 2.8)

set(CUDA_PATH "/usr/local/cuda" CACHE PATH "CUDA directory")
set(CUDA_TOOLKIT_ROOT_DIR "${CUDA_PATH}")
include(FindCUDA)

message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

# Default compilation for CUDA
find_path(NVCC nvcc${CMAKE_EXECUTABLE_SUFFIX}
          HINTS "${CUDA_PATH}/bin" ENV "CUDA_BIN_PATH")
if(CUDA_FOUND)
    set(HAVE_CUDA 1)
    set(API_DIR "cuda")
    include_directories(${CUDA_INCLUDE_DIRS})
else(CUDA_FOUND)
    message(FATAL_ERROR "NVIDIA CUDA Compiler is required")
endif(CUDA_FOUND)

set(COMMON_CXX_FLAGS "-std=c++11 -g")

if(CUDA_DEBUG)
    set(CUDA_NVCC_FLAGS ${COMMON_CXX_FLAGS} -DCUDARRAYS_DEBUG_CUDA=1)
endif(CUDA_DEBUG)

set(CMAKE_CXX_FLAGS_DEBUG   "${COMMON_CXX_FLAGS} -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${COMMON_CXX_FLAGS} -O3 -march=native")

set(GPU_ARCH "-arch" "sm_35"
             CACHE INTERNAL "")
set(CUDARRAYS_LIB "" CACHE INTERNAL "")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tests)
