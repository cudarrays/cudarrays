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

#include <iostream>

#include "cudarrays/storage.hpp"
#include "cudarrays/traits.hpp"

#include "gtest/gtest.h"

class traits_test :
    public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};


TEST_F(traits_test, array_dim_helper)
{
    ASSERT_EQ(cudarrays::array_extents_helper<int[1]>::type::dimensions, 1u);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1]>::type::static_dimensions, 1u);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1]>::type::dynamic_dimensions, 0u);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2]>::type::dimensions, 2u);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2]>::type::static_dimensions, 2u);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2]>::type::dynamic_dimensions, 0u);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2][3]>::type::dimensions, 3u);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2][3]>::type::static_dimensions, 3u);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2][3]>::type::dynamic_dimensions, 0u);

    ASSERT_EQ(cudarrays::array_extents_helper<int*>::type::dimensions, 1u);
    ASSERT_EQ(cudarrays::array_extents_helper<int*>::type::static_dimensions, 0u);
    ASSERT_EQ(cudarrays::array_extents_helper<int*>::type::dynamic_dimensions, 1u);
    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::dimensions, 2u);
    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::static_dimensions, 0u);
    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::dynamic_dimensions, 2u);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::dimensions, 3u);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::static_dimensions, 0u);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::dynamic_dimensions, 3u);

    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::dimensions, 3u);
    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::static_dimensions, 1u);
    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::dynamic_dimensions, 2u);
}

TEST_F(traits_test, array_extents_helper)
{
    ASSERT_EQ(cudarrays::array_extents_helper<int[1]>::type::get<0>(), 1u);

    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2]>::type::get<0>(), 1u);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2]>::type::get<1>(), 2u);

    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2][3]>::type::get<0>(), 1u);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2][3]>::type::get<1>(), 2u);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2][3]>::type::get<2>(), 3u);

    ASSERT_EQ(cudarrays::array_extents_helper<int*>::type::get<0>(), 0u);

    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::get<0>(), 0u);
    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::get<1>(), 0u);

    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::get<0>(), 0u);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::get<1>(), 0u);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::get<2>(), 0u);

    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::get<0>(), 0u);
    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::get<1>(), 0u);
    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::get<2>(), 3u);
}

TEST_F(traits_test, array_extents_helper2)
{
    ASSERT_EQ(cudarrays::array_extents_helper<int*>::type::get({1})[0], 1u);

    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::get({1, 2})[0], 1u);
    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::get({1, 2})[1], 2u);

    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::get({1, 2, 3})[0], 1u);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::get({1, 2, 3})[1], 2u);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::get({1, 2, 3})[2], 3u);

    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::get({1, 2})[0], 1u);
    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::get({1, 2})[1], 2u);
    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::get({1, 2})[2], 3u);
}

TEST_F(traits_test, array_offsets_helper)
{
    ASSERT_EQ(cudarrays::array_offsets_helper<int[1][2]>::type::get<0>(), 2u);
    ASSERT_EQ(cudarrays::array_offsets_helper<int[1][2][3]>::type::get<0>(), 6u);
    ASSERT_EQ(cudarrays::array_offsets_helper<int[1][2][3]>::type::get<1>(), 3u);

    ASSERT_EQ(cudarrays::array_offsets_helper<int **>::type::get<0>(), 0u);
    ASSERT_EQ(cudarrays::array_offsets_helper<int ***>::type::get<0>(), 0u);
    ASSERT_EQ(cudarrays::array_offsets_helper<int ***>::type::get<1>(), 0u);

    ASSERT_EQ(cudarrays::array_offsets_helper<int **[3]>::type::get<0>(), 0u);
    ASSERT_EQ(cudarrays::array_offsets_helper<int **[3]>::type::get<1>(), 3u);

    ASSERT_EQ(cudarrays::array_offsets_helper<int *[2][3]>::type::get<0>(), 6u);
    ASSERT_EQ(cudarrays::array_offsets_helper<int *[2][3]>::type::get<1>(), 3u);
}

TEST_F(traits_test, array_dim_reorder_helper)
{
    using traits = cudarrays::array_traits<int[1][2][3]>;

    using rmo_type = typename cudarrays::make_dim_order<traits::dimensions, cudarrays::layout::rmo>::seq_type;
    using cmo_type = typename cudarrays::make_dim_order<traits::dimensions, cudarrays::layout::cmo>::seq_type;

    using reorder_rmo_type =
        SEQ_REORDER( // User-provided dimension ordering
            traits::extents_type::extents_seq_type,
            rmo_type);

    using reorder_cmo_type =
        SEQ_REORDER( // User-provided dimension ordering
            traits::extents_type::extents_seq_type,
            cmo_type);

    using reorder_custom_type =
        SEQ_REORDER( // User-provided dimension ordering
            traits::extents_type::extents_seq_type,
            SEQ_WRAP(unsigned, cudarrays::layout::custom<1u, 2u, 0u>));

    ASSERT_EQ(reorder_rmo_type::as_array()[0], 1u);
    ASSERT_EQ(reorder_rmo_type::as_array()[1], 2u);
    ASSERT_EQ(reorder_rmo_type::as_array()[2], 3u);

    ASSERT_EQ(reorder_cmo_type::as_array()[0], 3u);
    ASSERT_EQ(reorder_cmo_type::as_array()[1], 2u);
    ASSERT_EQ(reorder_cmo_type::as_array()[2], 1u);

    ASSERT_EQ(reorder_custom_type::as_array()[0], 2u);
    ASSERT_EQ(reorder_custom_type::as_array()[1], 3u);
    ASSERT_EQ(reorder_custom_type::as_array()[2], 1u);
}
