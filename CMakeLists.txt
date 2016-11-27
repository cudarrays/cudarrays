project(cudatiles)
cmake_minimum_required(VERSION 3.0)

set(CUDA_PATH "/usr/local/cuda" CACHE PATH "CUDA directory")
set(CUDA_TOOLKIT_ROOT_DIR "${CUDA_PATH}")
find_package(CUDA REQUIRED)
find_package(OpenMP)

message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

message(STATUS "CUDA_LIBRARIES: ${CUDA_LIBRARIES}")

option(CUDA_DEBUG "Compile CUDA code with debug support" OFF)
option(ENABLE_GCOV "Compile with support for code coverage analysis" OFF)
option(PROFILE "Compile with support for code profiling" OFF)

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

if(CUDA_VERSION_MAJOR LESS "8")
    message(FATAL_ERROR "NVIDIA CUDA Compiler 8.0 or newer is required")
endif(CUDA_VERSION_MAJOR LESS "8")

set(COMMON_CXX_FLAGS "-Wall -Wextra -Wformat -Wformat-security -std=c++11 -I${CMAKE_BINARY_DIR}/include")

if(OPENMP_FOUND)
    set(COMMON_CXX_FLAGS "${COMMON_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OPENMP_FOUND)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

if(CUDA_DEBUG)
    set(CUDA_NVCC_FLAGS_DEBUG "-DCUDARRAYS_DEBUG_CUDA=1")
endif(CUDA_DEBUG)

if(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))
    set(CMAKE_CXX_FLAGS "${COMMON_CXX_FLAGS}")

    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -g -march=native")

    if(PROFILE)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fno-omit-frame-pointer")
    endif(PROFILE)

    set(CUDA_PROPAGATE_HOST_FLAGS ON)

    list(APPEND CUDA_NVCC_FLAGS "-keep" "--source-in-ptx" "--expt-relaxed-constexpr")

    set(CMAKE_EXE_LINKER_FLAGS_DEBUG   "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -g -O0")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} -g -O3 -march=native")

    if(ENABLE_GCOV)
        if(${CMAKE_BUILD_TYPE} MATCHES "Release")
            message(FATAL_ERROR "Code coverage analysis only supported in debug build type")
        endif(${CMAKE_BUILD_TYPE} MATCHES "Release")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} --coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
    endif(ENABLE_GCOV)
endif(("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") OR ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU"))

set(GPU_ARCH "-arch" "sm_35"
             CACHE INTERNAL "")
set(CUDARRAYS_LIB "" CACHE INTERNAL "")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tests)
