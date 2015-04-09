aux_source_directory(. LIB_SRC)
aux_source_directory(${CMAKE_SOURCE_DIR}/include LIB_INCLUDE)

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CUDA_PROPAGATE_HOST_FLAGS OFF)

option(CUDA_DEBUG "Compile CUDA code with debug support" OFF)
if(CMAKE_COMPILER_IS_GNUCC)
    set(CUDA_NVCC_FLAGS "-std=c++11" "-rdc" "true" "-g")
    list(APPEND CUDA_NVCC_FLAGS ${GPU_ARCH})
    if(CUDA_DEBUG)
        set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -G")
    endif(CUDA_DEBUG)
endif(CMAKE_COMPILER_IS_GNUCC)

cuda_add_library(${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays STATIC cuda_vars.cu ${LIB_SRC} ${LIB_INCLUDE})

set(CUDARRAYS_LIB ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays CACHE INTERNAL "")

install(TARGETS ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays
        ARCHIVE DESTINATION lib
        COMPONENT libraries)
