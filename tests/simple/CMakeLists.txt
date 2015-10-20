aux_source_directory(include LIB_INCLUDE)

set(LIB_INCLUDE ${CMAKE_SOURCE_DIR}/include)
include_directories(${LIB_INCLUDE})

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

option(CUDA_DEBUG "Compile CUDA code with debug support" ON)

list(APPEND CUDA_NVCC_FLAGS "--cudart" "static" "-L../../lib" "-lcudarrays")

if(CUDA_VERSION_MAJOR EQUAL "7" AND
   CUDA_VERSION_MINOR EQUAL "5")
   list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda")
endif(CUDA_VERSION_MAJOR EQUAL "7" AND
      CUDA_VERSION_MINOR EQUAL "5")

list(APPEND CUDA_NVCC_FLAGS ${GPU_ARCH})
if(CUDA_DEBUG)
    list(APPEND CUDA_NVCC_FLAGS "-G")
endif(CUDA_DEBUG)

cuda_add_executable(default default.cu ${LIB_INCLUDE})
target_link_libraries(default ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})
cuda_add_executable(inc inc.cu inc_kernel.cuh ${LIB_INCLUDE})
target_link_libraries(inc ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})
cuda_add_executable(matrixadd matrixadd.cu matrixadd_kernel.cuh ${LIB_INCLUDE})
target_link_libraries(matrixadd ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})
cuda_add_executable(matrixmul matrixmul.cu matrixmul_kernel.cuh ${LIB_INCLUDE})
target_link_libraries(matrixmul ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})
cuda_add_executable(saxpy saxpy.cu saxpy_kernel.cuh ${LIB_INCLUDE})
target_link_libraries(saxpy ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})
cuda_add_executable(static static.cu ${LIB_INCLUDE})
target_link_libraries(static ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})
cuda_add_executable(stencil stencil.cu stencil_kernel.cuh ${LIB_INCLUDE})
target_link_libraries(stencil ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})
cuda_add_executable(vecadd vecadd.cu vecadd_kernel.cuh ${LIB_INCLUDE})
target_link_libraries(vecadd ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})

set(COMMON_SRC common.hpp)

add_executable(alloc alloc.cpp ${COMMON_SRC} ${LIB_INCLUDE})
target_link_libraries(alloc ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})

add_executable(host host.cpp ${COMMON_SRC} ${LIB_INCLUDE})
target_link_libraries(host ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})

add_executable(storage_type storage_type.cpp ${COMMON_SRC} ${LIB_INCLUDE})
target_link_libraries(storage_type ${CMAKE_IMPORT_LIBRARY_PREFIX}cudarrays${CMAKE_IMPORT_LIBRARY_SUFFIX} ${CUDA_LIBRARIES})
