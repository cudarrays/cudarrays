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
    ASSERT_EQ(cudarrays::array_extents_helper<int>::type::dimensions, 0);

    ASSERT_EQ(cudarrays::array_extents_helper<int[1]>::type::dimensions, 1);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1]>::type::static_dimensions, 1);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1]>::type::dynamic_dimensions, 0);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2]>::type::dimensions, 2);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2]>::type::static_dimensions, 2);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2]>::type::dynamic_dimensions, 0);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2][3]>::type::dimensions, 3);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2][3]>::type::static_dimensions, 3);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2][3]>::type::dynamic_dimensions, 0);

    ASSERT_EQ(cudarrays::array_extents_helper<int*>::type::dimensions, 1);
    ASSERT_EQ(cudarrays::array_extents_helper<int*>::type::static_dimensions, 0);
    ASSERT_EQ(cudarrays::array_extents_helper<int*>::type::dynamic_dimensions, 1);
    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::dimensions, 2);
    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::static_dimensions, 0);
    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::dynamic_dimensions, 2);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::dimensions, 3);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::static_dimensions, 0);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::dynamic_dimensions, 3);

    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::dimensions, 3);
    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::static_dimensions, 1);
    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::dynamic_dimensions, 2);
}

TEST_F(traits_test, array_extents_helper)
{
    ASSERT_EQ(cudarrays::array_extents_helper<int[1]>::type::get<0>(), 1);

    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2]>::type::get<0>(), 1);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2]>::type::get<1>(), 2);

    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2][3]>::type::get<0>(), 1);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2][3]>::type::get<1>(), 2);
    ASSERT_EQ(cudarrays::array_extents_helper<int[1][2][3]>::type::get<2>(), 3);

    ASSERT_EQ(cudarrays::array_extents_helper<int*>::type::get<0>(), 0);

    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::get<0>(), 0);
    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::get<1>(), 0);

    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::get<0>(), 0);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::get<1>(), 0);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::get<2>(), 0);

    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::get<0>(), 0);
    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::get<1>(), 0);
    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::get<2>(), 3);
}

TEST_F(traits_test, array_extents_helper2)
{
    ASSERT_EQ(cudarrays::array_extents_helper<int*>::type::get({1})[0], 1);

    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::get({1, 2})[0], 1);
    ASSERT_EQ(cudarrays::array_extents_helper<int**>::type::get({1, 2})[1], 2);

    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::get({1, 2, 3})[0], 1);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::get({1, 2, 3})[1], 2);
    ASSERT_EQ(cudarrays::array_extents_helper<int***>::type::get({1, 2, 3})[2], 3);

    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::get({1, 2})[0], 1);
    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::get({1, 2})[1], 2);
    ASSERT_EQ(cudarrays::array_extents_helper<int**[3]>::type::get({1, 2})[2], 3);
}

TEST_F(traits_test, array_offsets_helper)
{
    ASSERT_EQ(cudarrays::array_offsets_helper<int[1][2]>::type::get<0>(), 2);
    ASSERT_EQ(cudarrays::array_offsets_helper<int[1][2][3]>::type::get<0>(), 6);
    ASSERT_EQ(cudarrays::array_offsets_helper<int[1][2][3]>::type::get<1>(), 3);

    ASSERT_EQ(cudarrays::array_offsets_helper<int **>::type::get<0>(), 0);
    ASSERT_EQ(cudarrays::array_offsets_helper<int ***>::type::get<0>(), 0);
    ASSERT_EQ(cudarrays::array_offsets_helper<int ***>::type::get<1>(), 0);

    ASSERT_EQ(cudarrays::array_offsets_helper<int **[3]>::type::get<0>(), 0);
    ASSERT_EQ(cudarrays::array_offsets_helper<int **[3]>::type::get<1>(), 3);

    ASSERT_EQ(cudarrays::array_offsets_helper<int *[2][3]>::type::get<0>(), 6);
    ASSERT_EQ(cudarrays::array_offsets_helper<int *[2][3]>::type::get<1>(), 3);
}
