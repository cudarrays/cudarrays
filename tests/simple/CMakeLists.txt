aux_source_directory(include LIB_INCLUDE)

set(LIB_INCLUDE ${CMAKE_SOURCE_DIR}/include)
include_directories(${LIB_INCLUDE})

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_SEPARABLE_COMPILATION ON)

option(CUDA_DEBUG "Compile CUDA code with debug support" OFF)
if(CMAKE_COMPILER_IS_GNUCC)
    set(CUDA_NVCC_FLAGS "-std=c++11" "--cudart" "static" "-g" "-L../../lib" "-lcudarrays")
    list(APPEND CUDA_NVCC_FLAGS ${GPU_ARCH})
    if(CUDA_DEBUG)
        list(APPEND CUDA_NVCC_FLAGS "-G")
    endif(CUDA_DEBUG)
endif(CMAKE_COMPILER_IS_GNUCC)

macro(cudarrays_compile in)
    set(extra_macro_args ${ARGN})
    add_custom_command(OUTPUT ${in}.o
                       COMMAND ${CUDA_NVCC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/${in} -dc -c -o ${in}.o ${CUDA_NVCC_FLAGS} -I${LIB_INCLUDE}
                       DEPENDS ${in} ${extra_macro_args})
endmacro(cudarrays_compile)

macro(cudarrays_link tname)
    set(extra_macro_args ${ARGN})
    add_custom_target(${tname}
                      ALL ${CUDA_NVCC_EXECUTABLE} ${tname}.cu.o -o ${tname} -link ${CUDA_NVCC_FLAGS}
                      DEPENDS ${tname}.cu.o cudarrays ${extra_macro_args})
endmacro(cudarrays_link)

cudarrays_compile(matrixadd.cu)
cudarrays_compile(matrixmul.cu)
cudarrays_compile(saxpy.cu)
cudarrays_compile(stencil.cu)
cudarrays_compile(vecadd.cu)
cudarrays_compile(default.cu)

cudarrays_link(matrixadd)
cudarrays_link(matrixmul)
cudarrays_link(saxpy)
cudarrays_link(stencil)
cudarrays_link(vecadd)
cudarrays_link(default)

set(COMMON_SRC common.hpp)

add_executable(alloc alloc.cpp ${COMMON_SRC} ${LIB_INCLUDE})
target_link_libraries(alloc ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})

add_executable(storage_type storage_type.cpp ${COMMON_SRC} ${LIB_INCLUDE})
target_link_libraries(storage_type ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})
