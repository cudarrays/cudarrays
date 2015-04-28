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

#include "cudarrays/utils.hpp"

#include "gtest/gtest.h"

class utils_test :
    public testing::Test {
protected:
    static void SetUpTestCase();
    static void TearDownTestCase();
};

void
utils_test::SetUpTestCase()
{
}

void
utils_test::TearDownTestCase()
{
}

template <typename T, typename U>
static unsigned do_loop_range(T min, U max)
{
    unsigned count = 0;
    for (auto i __attribute__((unused)) : utils::make_range(min, max)) {
        ++count;
    }
    return count;
}

TEST_F(utils_test, range)
{
    ASSERT_EQ(do_loop_range(-1, 2), 3u);
    ASSERT_EQ(do_loop_range(-1, 2u), 3u);
    ASSERT_EQ(do_loop_range(-1, 2l), 3u);

    ASSERT_EQ(do_loop_range(-1l, 2), 3u);
    ASSERT_EQ(do_loop_range(-1l, 2u), 3u);
    ASSERT_EQ(do_loop_range(-1l, 2l), 3u);
    ASSERT_EQ(do_loop_range(-1l, 2ul), 3u);
}

template <typename T>
static unsigned do_loop_range_low(T max)
{
    unsigned count = 0;
    for (auto i __attribute__((unused)) : utils::make_range(max)) {
        ++count;
    }
    return count;
}

TEST_F(utils_test, range_low)
{
    ASSERT_EQ(do_loop_range_low(5), 5u);
    ASSERT_EQ(do_loop_range_low(5u), 5u);
    ASSERT_EQ(do_loop_range_low(5l), 5u);
    ASSERT_EQ(do_loop_range_low(5ul), 5u);
}

TEST_F(utils_test, reorder_gather)
{
    using my_array = std::array<int, 5>;
    my_array a{0, 1, 2, 3, 4};
    ASSERT_EQ(utils::reorder_gather(a, a), a);

    my_array a_inv{4, 3, 2, 1, 0};
    ASSERT_EQ(utils::reorder_gather(a, a_inv), a_inv);

    my_array a_mix{1, 4, 0, 2, 3};
    ASSERT_EQ(utils::reorder_gather(a, a_mix), a_mix);
}

TEST_F(utils_test, reorder_scatter)
{
    using my_array = std::array<int, 5>;
    my_array a{0, 1, 2, 3, 4};
    ASSERT_EQ(utils::reorder_scatter(a, a), a);

    my_array a_inv{4, 3, 2, 1, 0};
    ASSERT_EQ(utils::reorder_scatter(a, a_inv), a_inv);

    my_array a_mix{1, 4, 0, 2, 3};
    my_array res_mix{2, 0, 3, 4, 1};
    ASSERT_EQ(utils::reorder_scatter(a, a_mix), res_mix);
}

TEST_F(utils_test, reorder_gather_static)
{
    using base2 = SEQ(0u, 1u);
    using reorder_2a = SEQ_REORDER(base2, SEQ(0u, 1u));
    ASSERT_EQ(reorder_2a::as_array()[0], 0u);
    ASSERT_EQ(reorder_2a::as_array()[1], 1u);

    using reorder_2b = SEQ_REORDER(base2, SEQ(1u, 0u));
    ASSERT_EQ(reorder_2b::as_array()[0], 1u);
    ASSERT_EQ(reorder_2b::as_array()[1], 0u);

    using base3 = SEQ(0u, 1, 2);
    using reorder_3a = SEQ_REORDER(base3, SEQ(0u, 1u, 2u));
    ASSERT_EQ(reorder_3a::as_array()[0], 0u);
    ASSERT_EQ(reorder_3a::as_array()[1], 1u);
    ASSERT_EQ(reorder_3a::as_array()[2], 2u);

    using reorder_3b = SEQ_REORDER(base3, SEQ(2u, 1u, 0u));
    ASSERT_EQ(reorder_3b::as_array()[0], 2u);
    ASSERT_EQ(reorder_3b::as_array()[1], 1u);
    ASSERT_EQ(reorder_3b::as_array()[2], 0u);

    using reorder_3c = SEQ_REORDER(base3, SEQ(1u, 2u, 0u));
    ASSERT_EQ(reorder_3c::as_array()[0], 1u);
    ASSERT_EQ(reorder_3c::as_array()[1], 2u);
    ASSERT_EQ(reorder_3c::as_array()[2], 0u);

    auto arr1 = base3::as_array();
    auto arr2 = reorder_3c::as_array();
    auto arr3 = utils::reorder_gather(arr1, std::array<unsigned, 3>{1u, 2u, 0u});
    ASSERT_EQ(arr2, arr3);
}
