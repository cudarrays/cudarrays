set(CUDARRAYS_HEADERS array.hpp
                      coherence.hpp
                      common.hpp
                      compiler.hpp
                      storage.hpp
                      storage_impl.hpp
                      dist_storage_reshape-block.hpp
                      dist_storage_reshape-cyclic.hpp
                      dist_storage_reshape-block_cyclic.hpp
                      dist_storage_replicated.hpp
                      dist_storage_vm.hpp
                      dynarray.hpp
                      env.hpp
                      gpu.cuh
                      indexing.hpp
                      integral_iterator.hpp
                      iterator.hpp
                      launch.hpp
                      launch_cpu.hpp
                      log.hpp
                      memory.hpp
                      trace.hpp
                      types.hpp
                      utils.hpp)

add_custom_target(headers SOURCES ${CUDARRAYS_HEADERS})

message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")

install(FILES ${CUDARRAYS_HEADERS}
        DESTINATION include/cudarrays
        COMPONENT headers)