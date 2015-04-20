include(FindThreads)

add_compile_options(-DCUDARRAYS_UNITTEST=1)

include_directories("../../include")

set (standalone_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/gtest/gtest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gtest/gtest-all.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTests.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/common.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/iterator.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/storage.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/traits.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/utils.cpp
)

add_executable(UnitTests ${standalone_SRC})

target_link_libraries(UnitTests ${CMAKE_THREAD_LIBS_INIT})


set (full_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/gtest/gtest.h
    ${CMAKE_CURRENT_SOURCE_DIR}/gtest/gtest-all.cc

    ${CMAKE_CURRENT_SOURCE_DIR}/UnitTests.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/lib_storage.cpp
)

add_executable(UnitTests_lib ${full_SRC})

target_link_libraries(UnitTests_lib ${CUDARRAYS_LIB} ${CMAKE_THREAD_LIBS_INIT})
