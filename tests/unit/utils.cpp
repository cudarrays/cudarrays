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
    unsigned count;
    count = 0;
    for (auto i : utils::make_range(min, max)) {
        ++count;
    }
    return count;
}

TEST_F(utils_test, range)
{
    ASSERT_EQ(do_loop_range(-1, 2), 3);
    ASSERT_EQ(do_loop_range(-1, 2u), 3);
    ASSERT_EQ(do_loop_range(-1, 2l), 3);

    ASSERT_EQ(do_loop_range(-1l, 2), 3);
    ASSERT_EQ(do_loop_range(-1l, 2u), 3);
    ASSERT_EQ(do_loop_range(-1l, 2l), 3);
    ASSERT_EQ(do_loop_range(-1l, 2ul), 3);
}

template <typename T>
static unsigned do_loop_range_low(T max)
{
    unsigned count;
    count = 0;
    for (auto i : utils::make_range(max)) {
        ++count;
    }
    return count;
}

TEST_F(utils_test, range_low)
{
    ASSERT_EQ(do_loop_range_low(5), 5);
    ASSERT_EQ(do_loop_range_low(5u), 5);
    ASSERT_EQ(do_loop_range_low(5l), 5);
    ASSERT_EQ(do_loop_range_low(5ul), 5);
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

template <unsigned _Val0, unsigned _Val1>
struct fake_type2 {
    const static unsigned Val0 = _Val0;
    const static unsigned Val1 = _Val1;

    static const std::array<unsigned, 2> as_array()
    {
        return {Val0, Val1};
    }
};

template <unsigned Val0, unsigned Val1>
const unsigned fake_type2<Val0, Val1>::Val0;
template <unsigned Val0, unsigned Val1>
const unsigned fake_type2<Val0, Val1>::Val1;

template <unsigned _Val0, unsigned _Val1, unsigned _Val2>
struct fake_type3 {
    const static unsigned Val0 = _Val0;
    const static unsigned Val1 = _Val1;
    const static unsigned Val2 = _Val2;

    static const std::array<unsigned, 3> as_array()
    {
        return {Val0, Val1, Val2};
    }
};
template <unsigned Val0, unsigned Val1, unsigned Val2>
const unsigned fake_type3<Val0, Val1, Val2>::Val0;
template <unsigned Val0, unsigned Val1, unsigned Val2>
const unsigned fake_type3<Val0, Val1, Val2>::Val1;
template <unsigned Val0, unsigned Val1, unsigned Val2>
const unsigned fake_type3<Val0, Val1, Val2>::Val2;

TEST_F(utils_test, reorder_gather_static)
{
    using base2 = fake_type2<0u, 1u>;
    using reorder_2a = utils::reorder_gather_static<2, unsigned, base2,
                                                                 fake_type2<0u, 1u> >;
    ASSERT_EQ(reorder_2a::type::Val0, 0u);
    ASSERT_EQ(reorder_2a::type::Val1, 1u);

    using reorder_2b = utils::reorder_gather_static<2, unsigned, base2,
                                                                  fake_type2<1u, 0u> >;
    ASSERT_EQ(reorder_2b::type::Val0, 1u);
    ASSERT_EQ(reorder_2b::type::Val1, 0u);

    using base3 = fake_type3<0u, 1u, 2u>;
    using reorder_3a = utils::reorder_gather_static<3, unsigned, base3,
                                                                 fake_type3<0u, 1u, 2u> >;
    ASSERT_EQ(reorder_3a::type::Val0, 0u);
    ASSERT_EQ(reorder_3a::type::Val1, 1u);
    ASSERT_EQ(reorder_3a::type::Val2, 2u);

    using reorder_3b = utils::reorder_gather_static<3, unsigned, base3,
                                                                 fake_type3<2u, 1u, 0u> >;
    ASSERT_EQ(reorder_3b::type::Val0, 2u);
    ASSERT_EQ(reorder_3b::type::Val1, 1u);
    ASSERT_EQ(reorder_3b::type::Val2, 0u);

    using reorder_3c = utils::reorder_gather_static<3, unsigned, base3,
                                                                 fake_type3<1u, 2u, 0u> >;
    ASSERT_EQ(reorder_3c::type::Val0, 1u);
    ASSERT_EQ(reorder_3c::type::Val1, 2u);
    ASSERT_EQ(reorder_3c::type::Val2, 0u);

    auto arr1 = base3::as_array();
    auto arr2 = reorder_3c::type::as_array();
    auto conf = reorder_3c::order_type::as_array();
    auto arr3 = utils::reorder_gather(arr1, conf);
    ASSERT_EQ(arr2, arr3);
}
