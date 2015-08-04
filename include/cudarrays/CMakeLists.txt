set(CUDARRAYS_BASE_HEADERS coherence.hpp
                      coherence.hpp
                      common.hpp
                      config.hpp
                      compiler.hpp
                      compute.hpp
                      dim_manager.hpp
                      dynarray.hpp
                      host.hpp
                      gpu.cuh
                      launch.hpp
                      launch_cpu.hpp
                      memory.hpp
                      storage.hpp
                      storage_impl.hpp
                      traits.hpp
                      trace.hpp
                      types.hpp
                      utils.hpp)

set(CUDARRAYS_DETAIL_COHERENCE_HEADERS
                      detail/coherence/default.hpp)

set(CUDARRAYS_DETAIL_DYNARRAY_HEADERS
                      detail/dynarray/base.hpp
                      detail/dynarray/helpers.hpp
                      detail/dynarray/indexing.hpp
                      detail/dynarray/iterator.hpp
                      detail/dynarray/storage_reshape-block.hpp
                      detail/dynarray/storage_reshape-cyclic.hpp
                      detail/dynarray/storage_reshape-block_cyclic.hpp
                      detail/dynarray/storage_replicated.hpp
                      detail/dynarray/storage_vm.hpp)

set(CUDARRAYS_DETAIL_UTILS_HEADERS
                      detail/utils/env.hpp
                      detail/utils/integral_iterator.hpp
                      detail/utils/log.hpp
                      detail/utils/misc.hpp
                      detail/utils/permute.hpp
                      detail/utils/seq.hpp
                      detail/utils/stl.hpp)

set(CUDARRAYS_HEADERS ${CUDARRAYS_BASE_HEADERS}
                      ${CUDARRAYS_DETAIL_COHERENCE_HEADERS}
                      ${CUDARRAYS_DETAIL_DYNARRAY_HEADERS}
                      ${CUDARRAYS_DETAIL_UTILS_HEADERS})

add_custom_target(headers SOURCES ${CUDARRAYS_HEADERS})

message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

install(FILES ${CUDARRAYS_BASE_HEADERS}
        DESTINATION include/cudarrays
        COMPONENT headers)

install(FILES ${CUDARRAYS_DETAIL_COHERENCE_HEADERS}
        DESTINATION include/cudarrays/detail/coherence
        COMPONENT headers)

install(FILES ${CUDARRAYS_DETAIL_DYNARRAY_HEADERS}
        DESTINATION include/cudarrays/detail/dynarray
        COMPONENT headers)

install(FILES ${CUDARRAYS_DETAIL_UTILS_HEADERS}
        DESTINATION include/cudarrays/detail/utils
        COMPONENT headers)
