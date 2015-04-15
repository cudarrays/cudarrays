set(CUDARRAYS_HEADERS array.hpp
                      coherence.hpp
                      common.hpp
                      config.hpp
                      compiler.hpp
                      compute.hpp
                      host.hpp
                      storage.hpp
                      storage_impl.hpp
                      detail/coherence/default.hpp
                      detail/dynarray/helpers.hpp
                      detail/dynarray/indexing.hpp
                      detail/dynarray/iterator.hpp
                      detail/dynarray/storage_reshape-block.hpp
                      detail/dynarray/storage_reshape-cyclic.hpp
                      detail/dynarray/storage_reshape-block_cyclic.hpp
                      detail/dynarray/storage_replicated.hpp
                      detail/dynarray/storage_vm.hpp
                      detail/utils/env.hpp
                      detail/utils/integral_iterator.hpp
                      detail/utils/log.hpp
                      detail/utils/misc.hpp
                      detail/utils/permute.hpp
                      detail/utils/stl.hpp
                      dynarray.hpp
                      gpu.cuh
                      launch.hpp
                      launch_cpu.hpp
                      memory.hpp
                      trace.hpp
                      types.hpp
                      utils.hpp)

add_custom_target(headers SOURCES ${CUDARRAYS_HEADERS})

message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

install(FILES ${CUDARRAYS_HEADERS}
        DESTINATION include/cudarrays
        COMPONENT headers)
