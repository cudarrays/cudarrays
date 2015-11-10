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
#include "cudarrays/array_traits.hpp"

#include "gtest/gtest.h"

class traits_test :
    public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(traits_test, dimensions)
{
    static_assert(cudarrays::array_traits<int[1]>::dimensions         == 1u, "Unexpected value");
    static_assert(cudarrays::array_traits<int[1]>::static_dimensions  == 1u, "Unexpected value");
    static_assert(cudarrays::array_traits<int[1]>::dynamic_dimensions == 0u, "Unexpected value");
    static_assert(cudarrays::array_traits<int[1][2]>::dimensions         == 2u, "Unexpected value");
    static_assert(cudarrays::array_traits<int[1][2]>::static_dimensions  == 2u, "Unexpected value");
    static_assert(cudarrays::array_traits<int[1][2]>::dynamic_dimensions == 0u, "Unexpected value");
    static_assert(cudarrays::array_traits<int[1][2][3]>::dimensions         == 3u, "Unexpected value");
    static_assert(cudarrays::array_traits<int[1][2][3]>::static_dimensions  == 3u, "Unexpected value");
    static_assert(cudarrays::array_traits<int[1][2][3]>::dynamic_dimensions == 0u, "Unexpected value");

    static_assert(cudarrays::array_traits<int*>::dimensions         == 1u, "Unexpected value");
    static_assert(cudarrays::array_traits<int*>::static_dimensions  == 0u, "Unexpected value");
    static_assert(cudarrays::array_traits<int*>::dynamic_dimensions == 1u, "Unexpected value");
    static_assert(cudarrays::array_traits<int**>::dimensions         == 2u, "Unexpected value");
    static_assert(cudarrays::array_traits<int**>::static_dimensions  == 0u, "Unexpected value");
    static_assert(cudarrays::array_traits<int**>::dynamic_dimensions == 2u, "Unexpected value");
    static_assert(cudarrays::array_traits<int***>::dimensions         == 3u, "Unexpected value");
    static_assert(cudarrays::array_traits<int***>::static_dimensions  == 0u, "Unexpected value");
    static_assert(cudarrays::array_traits<int***>::dynamic_dimensions == 3u, "Unexpected value");

    static_assert(cudarrays::array_traits<int**[3]>::dimensions         == 3u, "Unexpected value");
    static_assert(cudarrays::array_traits<int**[3]>::static_dimensions  == 1u, "Unexpected value");
    static_assert(cudarrays::array_traits<int**[3]>::dynamic_dimensions == 2u, "Unexpected value");
}

TEST_F(traits_test, array_extents_helper)
{
    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int[1]>::seq, 0), 1u);

    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int[1][2]>::seq, 0), 1u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int[1][2]>::seq, 1), 2u);

    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int[1][2][3]>::seq, 0), 1u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int[1][2][3]>::seq, 1), 2u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int[1][2][3]>::seq, 2), 3u);

    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int*>::seq, 0), 0u);

    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int**>::seq, 0), 0u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int**>::seq, 1), 0u);

    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int***>::seq, 0), 0u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int***>::seq, 1), 0u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int***>::seq, 2), 0u);

    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int**[3]>::seq, 0), 0u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int**[3]>::seq, 1), 0u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_extents_helper<int**[3]>::seq, 2), 3u);
}

TEST_F(traits_test, array_extents_helper2)
{
    using array1d = cudarrays::array_traits<int *>;
    ASSERT_EQ(array1d::make_extents({1})[0], 1u);

    using array2d = cudarrays::array_traits<int **>;
    ASSERT_EQ(array2d::make_extents({1, 2})[0], 1u);
    ASSERT_EQ(array2d::make_extents({1, 2})[1], 2u);

    using array3d = cudarrays::array_traits<int ***>;
    ASSERT_EQ(array3d::make_extents({1, 2, 3})[0], 1u);
    ASSERT_EQ(array3d::make_extents({1, 2, 3})[1], 2u);
    ASSERT_EQ(array3d::make_extents({1, 2, 3})[2], 3u);

    using array3d_hyb = cudarrays::array_traits<int **[3]>;
    ASSERT_EQ(array3d_hyb::make_extents({1, 2})[0], 1u);
    ASSERT_EQ(array3d_hyb::make_extents({1, 2})[1], 2u);
    ASSERT_EQ(array3d_hyb::make_extents({1, 2})[2], 3u);
}

TEST_F(traits_test, array_offsets_helper)
{
    using ss  = typename cudarrays::array_extents_helper<int [1][2]>::seq;
    using sss = typename cudarrays::array_extents_helper<int [1][2][3]>::seq;

    using dd  = typename cudarrays::array_extents_helper<int **>::seq;
    using ddd = typename cudarrays::array_extents_helper<int ***>::seq;

    using dds = typename cudarrays::array_extents_helper<int **[3]>::seq;
    using dss = typename cudarrays::array_extents_helper<int *[2][3]>::seq;

    std::cout << SEQ_SIZE(cudarrays::array_offsets_helper<ss>::seq) << "\n";

    ASSERT_EQ(SEQ_AT(cudarrays::array_offsets_helper<ss>::seq, 0), 2u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_offsets_helper<sss>::seq, 0), 6u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_offsets_helper<sss>::seq, 1), 3u);

    ASSERT_EQ(SEQ_AT(cudarrays::array_offsets_helper<dd>::seq, 0), 0u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_offsets_helper<ddd>::seq, 0), 0u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_offsets_helper<ddd>::seq, 1), 0u);

    ASSERT_EQ(SEQ_AT(cudarrays::array_offsets_helper<dds>::seq, 0), 0u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_offsets_helper<dds>::seq, 1), 3u);

    ASSERT_EQ(SEQ_AT(cudarrays::array_offsets_helper<dss>::seq, 0), 6u);
    ASSERT_EQ(SEQ_AT(cudarrays::array_offsets_helper<dss>::seq, 1), 3u);
}

TEST_F(traits_test, array_dim_reorder_helper)
{
    using traits = cudarrays::array_traits<int[1][2][3]>;

    using rmo_type = typename cudarrays::detail::make_dim_order<traits::dimensions, cudarrays::layout::rmo>::seq_type;
    using cmo_type = typename cudarrays::detail::make_dim_order<traits::dimensions, cudarrays::layout::cmo>::seq_type;

    using reorder_rmo_type =
        SEQ_REORDER( // User-provided dimension ordering
            traits::extents_seq,
            rmo_type);

    using reorder_cmo_type =
        SEQ_REORDER( // User-provided dimension ordering
            traits::extents_seq,
            cmo_type);

    using reorder_custom_type =
        SEQ_REORDER( // User-provided dimension ordering
            traits::extents_seq,
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

TEST_F(traits_test, storage_traits_extents)
{
    using rmo_traits = cudarrays::storage_traits<int[1][2][3],
                                                 cudarrays::layout::rmo,
                                                 cudarrays::noalign>;
    using cmo_traits = cudarrays::storage_traits<int[1][2][3],
                                                 cudarrays::layout::cmo,
                                                 cudarrays::noalign>;
    using custom_traits = cudarrays::storage_traits<int[1][2][3],
                                                    cudarrays::layout::custom<1u, 2u, 0u>,
                                                    cudarrays::noalign>;

    ASSERT_EQ(rmo_traits::extents_seq::as_array()[0], 1u);
    ASSERT_EQ(rmo_traits::extents_seq::as_array()[1], 2u);
    ASSERT_EQ(rmo_traits::extents_seq::as_array()[2], 3u);

    ASSERT_EQ(cmo_traits::extents_seq::as_array()[0], 3u);
    ASSERT_EQ(cmo_traits::extents_seq::as_array()[1], 2u);
    ASSERT_EQ(cmo_traits::extents_seq::as_array()[2], 1u);

    ASSERT_EQ(custom_traits::extents_seq::as_array()[0], 2u);
    ASSERT_EQ(custom_traits::extents_seq::as_array()[1], 3u);
    ASSERT_EQ(custom_traits::extents_seq::as_array()[2], 1u);
}

TEST_F(traits_test, storage_traits_extents2)
{
    using rmo_traits = cudarrays::storage_traits<int**[3],
                                                 cudarrays::layout::rmo,
                                                 cudarrays::noalign>;
    using cmo_traits = cudarrays::storage_traits<int**[3],
                                                 cudarrays::layout::cmo,
                                                 cudarrays::noalign>;
    using custom_traits = cudarrays::storage_traits<int**[3],
                                                    cudarrays::layout::custom<1u, 2u, 0u>,
                                                    cudarrays::noalign>;

    ASSERT_EQ(rmo_traits::extents_seq::as_array()[0], 0u);
    ASSERT_EQ(rmo_traits::extents_seq::as_array()[1], 0u);
    ASSERT_EQ(rmo_traits::extents_seq::as_array()[2], 3u);

    ASSERT_EQ(cmo_traits::extents_seq::as_array()[0], 0u);
    ASSERT_EQ(cmo_traits::extents_seq::as_array()[1], 0u);
    ASSERT_EQ(cmo_traits::extents_seq::as_array()[2], 0u);

    ASSERT_EQ(custom_traits::extents_seq::as_array()[0], 0u);
    ASSERT_EQ(custom_traits::extents_seq::as_array()[1], 0u);
    ASSERT_EQ(custom_traits::extents_seq::as_array()[2], 0u);
}

TEST_F(traits_test, storage_traits_extents_aligned)
{
    using rmo_traits = cudarrays::storage_traits<int[1][2][3],
                                                 cudarrays::layout::rmo,
                                                 cudarrays::align<4, 0>>;
    using cmo_traits = cudarrays::storage_traits<int[1][2][3],
                                                 cudarrays::layout::cmo,
                                                 cudarrays::align<4, 0>>;
    using custom_traits = cudarrays::storage_traits<int[1][2][3],
                                                    cudarrays::layout::custom<1u, 2u, 0u>,
                                                    cudarrays::align<4, 0>>;

    ASSERT_EQ(rmo_traits::extents_seq::as_array()[0], 1u);
    ASSERT_EQ(rmo_traits::extents_seq::as_array()[1], 2u);
    ASSERT_EQ(rmo_traits::extents_seq::as_array()[2], 4u);

    ASSERT_EQ(cmo_traits::extents_seq::as_array()[0], 3u);
    ASSERT_EQ(cmo_traits::extents_seq::as_array()[1], 2u);
    ASSERT_EQ(cmo_traits::extents_seq::as_array()[2], 4u);

    ASSERT_EQ(custom_traits::extents_seq::as_array()[0], 2u);
    ASSERT_EQ(custom_traits::extents_seq::as_array()[1], 3u);
    ASSERT_EQ(custom_traits::extents_seq::as_array()[2], 4u);
}

TEST_F(traits_test, storage_traits_extents2_aligned)
{
    using rmo_traits = cudarrays::storage_traits<int**[3],
                                                 cudarrays::layout::rmo,
                                                 cudarrays::align<4, 0>>;
    using cmo_traits = cudarrays::storage_traits<int**[3],
                                                 cudarrays::layout::cmo,
                                                 cudarrays::align<4, 0>>;
    using custom_traits = cudarrays::storage_traits<int**[3],
                                                    cudarrays::layout::custom<1u, 2u, 0u>,
                                                    cudarrays::align<4, 0>>;

    ASSERT_EQ(rmo_traits::extents_seq::as_array()[0], 0u);
    ASSERT_EQ(rmo_traits::extents_seq::as_array()[1], 0u);
    ASSERT_EQ(rmo_traits::extents_seq::as_array()[2], 4u);

    ASSERT_EQ(cmo_traits::extents_seq::as_array()[0], 0u);
    ASSERT_EQ(cmo_traits::extents_seq::as_array()[1], 0u);
    ASSERT_EQ(cmo_traits::extents_seq::as_array()[2], 0u);

    ASSERT_EQ(custom_traits::extents_seq::as_array()[0], 0u);
    ASSERT_EQ(custom_traits::extents_seq::as_array()[1], 0u);
    ASSERT_EQ(custom_traits::extents_seq::as_array()[2], 0u);
}


TEST_F(traits_test, storage_traits_offset)
{
    using rmo_traits = cudarrays::storage_traits<int[1][2][3],
                                                 cudarrays::layout::rmo,
                                                 cudarrays::noalign>;
    using cmo_traits = cudarrays::storage_traits<int[1][2][3],
                                                 cudarrays::layout::cmo,
                                                 cudarrays::noalign>;
    using custom_traits = cudarrays::storage_traits<int[1][2][3],
                                                    cudarrays::layout::custom<1u, 2u, 0u>,
                                                    cudarrays::noalign>;

    ASSERT_EQ(rmo_traits::offsets_seq::as_array()[0], 6u);
    ASSERT_EQ(rmo_traits::offsets_seq::as_array()[1], 3u);

    ASSERT_EQ(cmo_traits::offsets_seq::as_array()[0], 2u);
    ASSERT_EQ(cmo_traits::offsets_seq::as_array()[1], 1u);

    ASSERT_EQ(custom_traits::offsets_seq::as_array()[0], 3u);
    ASSERT_EQ(custom_traits::offsets_seq::as_array()[1], 1u);
}

TEST_F(traits_test, storage_traits_offset2)
{
    using rmo_traits = cudarrays::storage_traits<int**[3],
                                                 cudarrays::layout::rmo,
                                                 cudarrays::noalign>;
    using cmo_traits = cudarrays::storage_traits<int**[3],
                                                 cudarrays::layout::cmo,
                                                 cudarrays::noalign>;
    using custom_traits = cudarrays::storage_traits<int**[3],
                                                    cudarrays::layout::custom<1u, 2u, 0u>,
                                                    cudarrays::noalign>;

    ASSERT_EQ(rmo_traits::offsets_seq::as_array()[0], 0u);
    ASSERT_EQ(rmo_traits::offsets_seq::as_array()[1], 3u);

    ASSERT_EQ(cmo_traits::offsets_seq::as_array()[0], 0u);
    ASSERT_EQ(cmo_traits::offsets_seq::as_array()[1], 0u);

    ASSERT_EQ(custom_traits::offsets_seq::as_array()[0], 0u);
    ASSERT_EQ(custom_traits::offsets_seq::as_array()[1], 0u);
}

TEST_F(traits_test, dist_storage_traits_partition)
{
    using rmo_traits1 = cudarrays::dist_storage_traits<int[1][2][3],
                                                       cudarrays::layout::rmo,
                                                       cudarrays::noalign,
                                                       cudarrays::storage_conf<cudarrays::detail::storage_tag::VM,
                                                                               cudarrays::partition(0b100)>>;
    using rmo_traits2 = cudarrays::dist_storage_traits<int[1][2][3],
                                                       cudarrays::layout::rmo,
                                                       cudarrays::noalign,
                                                       cudarrays::storage_conf<cudarrays::detail::storage_tag::VM,
                                                                               cudarrays::partition(0b001)>>;

    using cmo_traits1 = cudarrays::dist_storage_traits<int[1][2][3],
                                                       cudarrays::layout::cmo,
                                                       cudarrays::noalign,
                                                       cudarrays::storage_conf<cudarrays::detail::storage_tag::VM,
                                                                               cudarrays::partition(0b100)>>;
    using cmo_traits2 = cudarrays::dist_storage_traits<int[1][2][3],
                                                       cudarrays::layout::cmo,
                                                       cudarrays::noalign,
                                                       cudarrays::storage_conf<cudarrays::detail::storage_tag::VM,
                                                                               cudarrays::partition(0b001)>>;

    ASSERT_EQ(rmo_traits1::partition_value, cudarrays::partition(0b100));
    ASSERT_EQ(rmo_traits2::partition_value, cudarrays::partition(0b001));

    ASSERT_EQ(cmo_traits1::partition_value, cudarrays::partition(0b001));
    ASSERT_EQ(cmo_traits2::partition_value, cudarrays::partition(0b100));
}
