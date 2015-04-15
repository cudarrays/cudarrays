/*
 * CUDArrays is a library for easy multi-GPU program development.
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2013-2015 Barcelona Supercomputing Center and
 *                         University of Illinois
 *
 *  Developed by: Javier Cabezas <javier.cabezas@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

#include <array>
#include <iostream>

#include "common.hpp"

#include "cudarrays/dynarray.hpp"

#include "gtest/gtest.h"

class lib_storage_test :
    public testing::Test {
protected:
    static void SetUpTestCase();
    static void TearDownTestCase();
};

void
lib_storage_test::SetUpTestCase()
{
    cudarrays::init_lib();
}

void
lib_storage_test::TearDownTestCase()
{
    cudarrays::fini_lib();
}

#if 0
template <unsigned Dims>
using extents = std::array<cudarrays::array_size_t, Dims>;

#define do_host_alloc(P)                                                              \
{                                                                                     \
    using my_dims  = cudarrays::dim_manager<float, 3>;                                \
    using my_array = cudarrays::host_storage<float>;                                  \
                                                                                      \
    static const cudarrays::array_size_t Z = 3;                                       \
    static const cudarrays::array_size_t Y = 5;                                       \
    static const cudarrays::array_size_t X = 7;                                       \
                                                                                      \
    using extents_type = std::array<cudarrays::array_size_t, 3>;                      \
                                                                                      \
    float regular[Z][Y][X];                                                           \
                                                                                      \
    my_dims  D1{extents_type{Z, Y, X}, cudarrays::align_t{}};                         \
    my_array A1{extents_type{Z, Y, X}, cudarrays::align_t{}};                         \
                                                                                      \
    for (auto i : make_range(Z)) {                                                    \
        for (auto j : make_range(Y)) {                                                \
            for (auto k : make_range(X)) {                                            \
                A1.access_pos(i, j, k) = float(i) + float(j) + float(k);              \
                regular[i][j][k]       = float(i) + float(j) + float(k);              \
            }                                                                         \
        }                                                                             \
    }                                                                                 \
                                                                                      \
    float (&internal)[Z][Y][X] = *(float(*)[Z][Y][X]) A1.get_host_storage().get_addr(); \
                                                                                      \
    for (auto i : make_range(Z)) {                                                    \
        for (auto j : make_range(Y)) {                                                \
            for (auto k : make_range(X)) {                                            \
                ASSERT_EQ(regular[i][j][k], internal[i][j][k]);                       \
            }                                                                         \
        }                                                                             \
    }                                                                                 \
}

TEST_F(lib_storage_test, host_replicated)
{
    do_host_alloc(cudarrays::replicate);
}

TEST_F(lib_storage_test, host_reshape_block)
{
    do_host_alloc(cudarrays::reshape_block);
}

TEST_F(lib_storage_test, host_reshape_block_cyclic)
{
    do_host_alloc(cudarrays::reshape_block_cyclic);
}

TEST_F(lib_storage_test, host_reshape_cyclic)
{
    do_host_alloc(cudarrays::reshape_cyclic);
}

TEST_F(lib_storage_test, host_vm)
{
    do_host_alloc(cudarrays::vm);
}
#endif
