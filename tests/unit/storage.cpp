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

#include "common.hpp"

#include "cudarrays/storage.hpp"
#include "cudarrays/detail/dynarray/storage_vm.hpp"

#include "gtest/gtest.h"

class storage_test :
    public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

TEST_F(storage_test, permute_indices1)
{
    using conf_identity = SEQ_WRAP(unsigned, cudarrays::layout::custom<0u, 1u, 2u>);
    using conf_reverse  = SEQ_WRAP(unsigned, cudarrays::layout::custom<2u, 1u, 0u>);
    using conf_mix      = SEQ_WRAP(unsigned, cudarrays::layout::custom<1u, 2u, 0u>);

    using permuter_identity = utils::permuter<conf_identity>;
    using permuter_reverse  = utils::permuter<conf_reverse>;
    using permuter_mix      = utils::permuter<conf_mix>;

    ASSERT_EQ(permuter_identity::template select<0>(0, 1, 2), 0);
    ASSERT_EQ(permuter_identity::template select<1>(0, 1, 2), 1);
    ASSERT_EQ(permuter_identity::template select<2>(0, 1, 2), 2);

    ASSERT_EQ(permuter_reverse::template select<0>(0, 1, 2), 2);
    ASSERT_EQ(permuter_reverse::template select<1>(0, 1, 2), 1);
    ASSERT_EQ(permuter_reverse::template select<2>(0, 1, 2), 0);

    ASSERT_EQ(permuter_mix::template select<0>(0, 1, 2), 2);
    ASSERT_EQ(permuter_mix::template select<1>(0, 1, 2), 0);
    ASSERT_EQ(permuter_mix::template select<2>(0, 1, 2), 1);
}

TEST_F(storage_test, permute_indices2)
{
    using conf_rmo = typename cudarrays::detail::make_dim_order<3, cudarrays::layout::rmo>::seq_type;
    using conf_cmo = typename cudarrays::detail::make_dim_order<3, cudarrays::layout::cmo>::seq_type;

    using permuter_rmo = utils::permuter<conf_rmo>;
    using permuter_cmo = utils::permuter<conf_cmo>;

    ASSERT_EQ(permuter_rmo::template select<0>(0, 1, 2), 0);
    ASSERT_EQ(permuter_rmo::template select<1>(0, 1, 2), 1);
    ASSERT_EQ(permuter_rmo::template select<2>(0, 1, 2), 2);

    ASSERT_EQ(permuter_cmo::template select<0>(0, 1, 2), 2);
    ASSERT_EQ(permuter_cmo::template select<1>(0, 1, 2), 1);
    ASSERT_EQ(permuter_cmo::template select<2>(0, 1, 2), 0);
}

TEST_F(storage_test, reorder)
{
    using conf_identity = SEQ_WRAP(unsigned, cudarrays::layout::custom<0u, 1u, 2u>);
    using conf_reverse  = SEQ_WRAP(unsigned, cudarrays::layout::custom<2u, 1u, 0u>);
    using conf_mix      = SEQ_WRAP(unsigned, cudarrays::layout::custom<1u, 2u, 0u>);

    using permuter_identity = utils::permuter<conf_identity>;
    using permuter_reverse  = utils::permuter<conf_reverse>;
    using permuter_mix      = utils::permuter<conf_mix>;

    auto identity = permuter_identity::reorder(std::array<unsigned, 3>{0, 1, 2});
    auto reverse  = permuter_reverse::reorder(std::array<unsigned, 3>{0, 1, 2});
    auto mix      = permuter_mix::reorder(std::array<unsigned, 3>{0, 1, 2});

    ASSERT_EQ(identity[0], 0u);
    ASSERT_EQ(identity[1], 1u);
    ASSERT_EQ(identity[2], 2u);

    ASSERT_EQ(reverse[0], 2u);
    ASSERT_EQ(reverse[1], 1u);
    ASSERT_EQ(reverse[2], 0u);

    ASSERT_EQ(mix[0], 1u);
    ASSERT_EQ(mix[1], 2u);
    ASSERT_EQ(mix[2], 0u);
}

TEST_F(storage_test, bitseq_seq)
{
    using T0 = typename utils::bitset_to_seq<0b0, 1>::type;
    ASSERT_EQ(utils::seq_to_bitset<T0>::value, 0b0u);
    using T1 = typename utils::bitset_to_seq<0b1, 1>::type;
    ASSERT_EQ(utils::seq_to_bitset<T1>::value, 0b1u);

    using T00 = typename utils::bitset_to_seq<0b00, 2>::type;
    ASSERT_EQ(utils::seq_to_bitset<T00>::value, 0b00u);
    using T01 = typename utils::bitset_to_seq<0b01, 2>::type;
    ASSERT_EQ(utils::seq_to_bitset<T01>::value, 0b01u);
    using T10 = typename utils::bitset_to_seq<0b10, 2>::type;
    ASSERT_EQ(utils::seq_to_bitset<T10>::value, 0b10u);
    using T11 = typename utils::bitset_to_seq<0b11, 2>::type;
    ASSERT_EQ(utils::seq_to_bitset<T11>::value, 0b11u);

    using T000 = typename utils::bitset_to_seq<0b000, 3>::type;
    ASSERT_EQ(utils::seq_to_bitset<T000>::value, 0b000u);
    using T001 = typename utils::bitset_to_seq<0b001, 3>::type;
    ASSERT_EQ(utils::seq_to_bitset<T001>::value, 0b001u);
    using T010 = typename utils::bitset_to_seq<0b010, 3>::type;
    ASSERT_EQ(utils::seq_to_bitset<T010>::value, 0b010u);
    using T011 = typename utils::bitset_to_seq<0b011, 3>::type;
    ASSERT_EQ(utils::seq_to_bitset<T011>::value, 0b011u);
    using T100 = typename utils::bitset_to_seq<0b100, 3>::type;
    ASSERT_EQ(utils::seq_to_bitset<T100>::value, 0b100u);
    using T101 = typename utils::bitset_to_seq<0b101, 3>::type;
    ASSERT_EQ(utils::seq_to_bitset<T101>::value, 0b101u);
    using T110 = typename utils::bitset_to_seq<0b110, 3>::type;
    ASSERT_EQ(utils::seq_to_bitset<T110>::value, 0b110u);
    using T111 = typename utils::bitset_to_seq<0b111, 3>::type;
    ASSERT_EQ(utils::seq_to_bitset<T111>::value, 0b111u);
}

TEST_F(storage_test, part_helper_none)
{
    using part_helper_1_none = typename cudarrays::storage_part_helper<cudarrays::partition::NONE, 1>::type;
    using part_helper_2_none = typename cudarrays::storage_part_helper<cudarrays::partition::NONE, 2>::type;
    using part_helper_3_none = typename cudarrays::storage_part_helper<cudarrays::partition::NONE, 3>::type;

    ASSERT_EQ(part_helper_1_none::as_array().size(), 1u);
    ASSERT_EQ(part_helper_2_none::as_array().size(), 2u);
    ASSERT_EQ(part_helper_3_none::as_array().size(), 3u);
}

TEST_F(storage_test, part_helper_single)
{
    using part_helper_1_x = typename cudarrays::storage_part_helper<cudarrays::partition::X, 1>::type;
    using part_helper_2_x = typename cudarrays::storage_part_helper<cudarrays::partition::X, 2>::type;
    using part_helper_3_x = typename cudarrays::storage_part_helper<cudarrays::partition::X, 3>::type;

    ASSERT_EQ(part_helper_1_x::as_array()[0], true);
    ASSERT_EQ(part_helper_2_x::as_array()[0], false);
    ASSERT_EQ(part_helper_2_x::as_array()[1], true);
    ASSERT_EQ(part_helper_3_x::as_array()[0], false);
    ASSERT_EQ(part_helper_3_x::as_array()[1], false);
    ASSERT_EQ(part_helper_3_x::as_array()[2], true);

    using part_helper_2_y = typename cudarrays::storage_part_helper<cudarrays::partition::Y, 2>::type;
    using part_helper_3_y = typename cudarrays::storage_part_helper<cudarrays::partition::Y, 3>::type;

    ASSERT_EQ(part_helper_2_y::as_array()[0], true);
    ASSERT_EQ(part_helper_2_y::as_array()[1], false);
    ASSERT_EQ(part_helper_3_y::as_array()[0], false);
    ASSERT_EQ(part_helper_3_y::as_array()[1], true);
    ASSERT_EQ(part_helper_3_y::as_array()[2], false);

    using part_helper_3_z = typename cudarrays::storage_part_helper<cudarrays::partition::Z, 3>::type;

    ASSERT_EQ(part_helper_3_z::as_array()[0], true);
    ASSERT_EQ(part_helper_3_z::as_array()[1], false);
    ASSERT_EQ(part_helper_3_z::as_array()[2], false);
}

template <unsigned Dims>
using extents = cudarrays::extents<Dims>;

template <typename Align>
using my_2d_manager = cudarrays::dim_manager<float, Align, 2>;

TEST_F(storage_test, dim_manager)
{
    // Small extents
    extents<2> extents_small{3, 5};
    my_2d_manager<cudarrays::noalign> mgr1{extents_small};

    ASSERT_EQ(mgr1.dims()[0], extents_small[0]);
    ASSERT_EQ(mgr1.dims()[1], extents_small[1]);
    ASSERT_EQ(mgr1.dims_align()[0], extents_small[0]);
    ASSERT_EQ(mgr1.dims_align()[1], extents_small[1]);

    using align_16 = cudarrays::align<16>;

    my_2d_manager<align_16> mgr2{extents_small};

    ASSERT_EQ(mgr2.dims()[0], extents_small[0]);
    ASSERT_EQ(mgr2.dims()[1], extents_small[1]);
    ASSERT_EQ(mgr2.dims_align()[0], extents_small[0]);
    ASSERT_EQ(mgr2.dims_align()[1], align_16::alignment);

    // Big extents
    extents<2> extents_big{247, 251};
    my_2d_manager<align_16> mgr3{extents_big};

    ASSERT_EQ(mgr3.dims()[0], extents_big[0]);
    ASSERT_EQ(mgr3.dims()[1], extents_big[1]);
    ASSERT_EQ(mgr3.dims_align()[0], extents_big[0]);
    ASSERT_GT(mgr3.dims_align()[1], extents_big[1]);
    ASSERT_EQ(mgr3.dims_align()[1] % align_16::alignment, 0u);
}

TEST_F(storage_test, dim_manager_offset)
{
    // Small extents
    extents<2> extents{3, 7};

    using align_4 = cudarrays::align<4>;
    my_2d_manager<align_4> mgr1{extents};

    ASSERT_EQ(mgr1.offset(), 0u);

    using align_4_2 = cudarrays::align<4, 2>;
    my_2d_manager<align_4_2> mgr2{extents};

    ASSERT_EQ(mgr2.offset(), 2u);

    using align_4_5 = cudarrays::align<4, 5>;
    my_2d_manager<align_4_5> mgr3{extents};

    ASSERT_EQ(mgr3.offset(), 3u);
}

template <typename Align>
using my_3d_manager = cudarrays::dim_manager<float, Align, 3>;

TEST_F(storage_test, dim_manager_impl_offset)
{
    // Small extents
    extents<3> extents{5, 3, 7};

    using align_4 = cudarrays::align<4>;
    my_3d_manager<align_4> mgr1{extents};

    ASSERT_EQ(mgr1.dims()[2],       7u);
    ASSERT_EQ(mgr1.dims_align()[2], 8u);
    ASSERT_EQ(mgr1.dims()[1],       3u);
    ASSERT_EQ(mgr1.dims_align()[1], 3u);
    ASSERT_EQ(mgr1.dims()[0],       5u);
    ASSERT_EQ(mgr1.dims_align()[0], 5u);

    ASSERT_EQ(mgr1.get_strides()[1], mgr1.dims_align()[2]);
}

TEST_F(storage_test, dim_manager_get_dim)
{
    extents<2> extents{3, 5};
    my_2d_manager<cudarrays::noalign> mgr1{extents};

    ASSERT_EQ(mgr1.dims()[0], mgr1.dim(0));
    ASSERT_EQ(mgr1.dims()[1], mgr1.dim(1));
    ASSERT_EQ(mgr1.dims_align()[0], mgr1.dim_align(0));
    ASSERT_EQ(mgr1.dims_align()[1], mgr1.dim_align(1));
}

template <typename Align>
using my_storage = cudarrays::host_storage<cudarrays::storage_traits<float *, cudarrays::layout::rmo, Align>>;

TEST_F(storage_test, host_storage)
{
    my_storage<cudarrays::noalign> mgr{};

    mgr.alloc(10);
    ASSERT_EQ(mgr.addr(), mgr.base_addr());

    my_storage<cudarrays::align<2, 1>> mgr2{};

    mgr.alloc(10);
    ASSERT_NE(mgr2.addr(), mgr2.base_addr());
    ASSERT_EQ(mgr2.addr() - mgr2.base_addr(), 1);
}

TEST_F(storage_test, vm_page_allocator1)
{
    using my_page_allocator = cudarrays::page_allocator<3>;

    extents<3>dims  { 5, 7, 4 };
    extents<3>align = dims;

    std::array<unsigned, 3> arrayDimToGpus {  0, 2, 1 };

    extents<3> local { 5, 3, 2 };

    my_page_allocator page_allocator{6, dims, align, local, arrayDimToGpus, 3};

    ASSERT_EQ(page_allocator.dims_[0], 5u);
    ASSERT_EQ(page_allocator.dims_[1], 7u);
    ASSERT_EQ(page_allocator.dims_[2], 4u);

    auto ret = page_allocator.advance();

    ASSERT_EQ(page_allocator.idx_[0], 0u);
    ASSERT_EQ(page_allocator.idx_[1], 0u);
    ASSERT_EQ(page_allocator.idx_[2], 3u);

    ret = page_allocator.advance();

    ASSERT_EQ(page_allocator.idx_[0], 0u);
    ASSERT_EQ(page_allocator.idx_[1], 1u);
    ASSERT_EQ(page_allocator.idx_[2], 2u);

    ret = page_allocator.advance();

    ASSERT_EQ(page_allocator.idx_[0], 0u);
    ASSERT_EQ(page_allocator.idx_[1], 2u);
    ASSERT_EQ(page_allocator.idx_[2], 1u);

    ret = page_allocator.advance();

    ASSERT_EQ(page_allocator.idx_[0], 0u);
    ASSERT_EQ(page_allocator.idx_[1], 3u);
    ASSERT_EQ(page_allocator.idx_[2], 0u);

    while (ret.first == false) {
        ret = page_allocator.advance();
    }

    ASSERT_EQ(page_allocator.idx_[0], 5u);
    ASSERT_EQ(page_allocator.idx_[1], 0u);
    ASSERT_EQ(page_allocator.idx_[2], 0u);
}

TEST_F(storage_test, vm_page_allocator2)
{
    using my_page_allocator = cudarrays::page_allocator<3>;

    extents<3> dims  { 5, 7, 4 };
    extents<3> align = dims;

    std::array<unsigned, 3> arrayDimToGpus {  0, 2, 1 };

    extents<3> local { 5, 3, 2 };

    my_page_allocator page_allocator{6, dims, align, local, arrayDimToGpus, 4};

    ASSERT_EQ(page_allocator.dims_[0], 5u);
    ASSERT_EQ(page_allocator.dims_[1], 7u);
    ASSERT_EQ(page_allocator.dims_[2], 4u);

    page_allocator.advance();

    ASSERT_EQ(page_allocator.idx_[0], 0u);
    ASSERT_EQ(page_allocator.idx_[1], 1u);
    ASSERT_EQ(page_allocator.idx_[2], 0u);

    page_allocator.advance();
    ASSERT_EQ(page_allocator.idx_[0], 0u);
    ASSERT_EQ(page_allocator.idx_[1], 2u);
    ASSERT_EQ(page_allocator.idx_[2], 0u);

    page_allocator.advance();

    ASSERT_EQ(page_allocator.idx_[0], 0u);
    ASSERT_EQ(page_allocator.idx_[1], 3u);
    ASSERT_EQ(page_allocator.idx_[2], 0u);

    page_allocator.advance();

    ASSERT_EQ(page_allocator.idx_[0], 0u);
    ASSERT_EQ(page_allocator.idx_[1], 4u);
    ASSERT_EQ(page_allocator.idx_[2], 0u);

    std::pair<bool, typename my_page_allocator::page_stats> ret;
    do {
        ret = page_allocator.advance();
    } while (ret.first == false);

    ASSERT_EQ(page_allocator.idx_[0], 5u);
    ASSERT_EQ(page_allocator.idx_[1], 0u);
    ASSERT_EQ(page_allocator.idx_[2], 0u);

    ASSERT_EQ(page_allocator.get_imbalance_ratio(), 0.5);
}


template <unsigned Dims>
static
void gpu_grid_conf(const cudarrays::compute_conf<Dims> &conf, const std::array<unsigned, Dims> &result)
{
    auto grid = cudarrays::helper_distribution_get_gpu_grid(conf);

    ASSERT_EQ(grid.size(), Dims);

    for (auto i : utils::make_range(Dims)) {
        ASSERT_EQ(grid[i], result[i]);
    }
}

TEST_F(storage_test, gpu_grid)
{
    //
    // 1D decompositions
    //
    gpu_grid_conf<1>({cudarrays::compute::none, 12}, {1});
    gpu_grid_conf<1>({cudarrays::compute::x, 12},    {12});

    //
    // 2D decompositions
    //
    gpu_grid_conf<2>({cudarrays::compute::none, 12}, {1, 1});
    gpu_grid_conf<2>({cudarrays::compute::x, 12},    {1, 12});
    gpu_grid_conf<2>({cudarrays::compute::y, 12},    {12, 1});
    gpu_grid_conf<2>({cudarrays::compute::xy, 12},   {6, 2});

    //
    // 3D decompositions
    //
    gpu_grid_conf<3>({cudarrays::compute::none, 12}, {1, 1, 1});
    gpu_grid_conf<3>({cudarrays::compute::x, 12},    {1, 1, 12});
    gpu_grid_conf<3>({cudarrays::compute::y, 12},    {1, 12, 1});
    gpu_grid_conf<3>({cudarrays::compute::z, 12},    {12, 1, 1});
    gpu_grid_conf<3>({cudarrays::compute::xy, 12},   {1, 6, 2});
    gpu_grid_conf<3>({cudarrays::compute::xz, 12},   {6, 1, 2});
    gpu_grid_conf<3>({cudarrays::compute::yz, 12},   {6, 2, 1});
    gpu_grid_conf<3>({cudarrays::compute::xyz, 12},  {3, 2, 2});
}

template <unsigned DimsComp, unsigned Dims>
static inline
void array_grid_conf(const cudarrays::compute_mapping<DimsComp, Dims> &mapping,
                     const std::array<unsigned, Dims> &result)
{
    auto gpuGrid           = cudarrays::helper_distribution_get_gpu_grid(mapping.comp);
    auto arrayDimToCompDim = mapping.get_array_to_comp(); // TODO: check this function

    auto arrayGrid = cudarrays::helper_distribution_get_array_grid(gpuGrid, arrayDimToCompDim);

    for (auto i : utils::make_range(Dims)) {
        if (arrayGrid[i] != result[i]) abort();
        ASSERT_EQ(arrayGrid[i], result[i]);
    }
}

TEST_F(storage_test, array_grid)
{
    //
    // 1D decompositions
    //
    array_grid_conf<1, 1>({{cudarrays::compute::none, 12}, {-1}},
                          {1});
    array_grid_conf<1, 1>({{cudarrays::compute::x, 12}, {0}},
                          {12});

    //
    // 2D decompositions
    //
    array_grid_conf<2, 2>({{cudarrays::compute::none, 12}, {-1, -1}},
                          {1, 1});
    array_grid_conf<2, 2>({{cudarrays::compute::x, 12}, {-1, 0}},
                          {1, 12});
    array_grid_conf<2, 2>({{cudarrays::compute::y, 12}, {1, -1}},
                          {12, 1});
    array_grid_conf<2, 2>({{cudarrays::compute::xy, 12}, {1, 0}},
                          {6, 2});
    array_grid_conf<2, 2>({{cudarrays::compute::xy, 12}, {0, 1}},
                          {2, 6});

    //
    // 3D decompositions
    //
    array_grid_conf<3, 3>({{cudarrays::compute::none, 12}, {-1, -1, -1}},
                          {1, 1, 1});
    array_grid_conf<3, 3>({{cudarrays::compute::x, 12}, {-1, -1, 0}},
                          {1, 1, 12});
    array_grid_conf<3, 3>({{cudarrays::compute::y, 12}, {-1, 1, -1}},
                          {1, 12, 1});
    array_grid_conf<3, 3>({{cudarrays::compute::z, 12}, {2, -1, -1}},
                          {12, 1, 1});
    array_grid_conf<3, 3>({{cudarrays::compute::xy, 12}, {-1, 1, 0}},
                          {1, 6, 2});
    array_grid_conf<3, 3>({{cudarrays::compute::xy, 12}, {-1, 0, 1}},
                          {1, 2, 6});
    array_grid_conf<3, 3>({{cudarrays::compute::xz, 12}, {2, -1, 0}},
                          {6, 1, 2});
    array_grid_conf<3, 3>({{cudarrays::compute::xz, 12}, {0, -1, 2}},
                          {2, 1, 6});
    array_grid_conf<3, 3>({{cudarrays::compute::yz, 12}, {2, 1, -1}},
                          {6, 2, 1});
    array_grid_conf<3, 3>({{cudarrays::compute::yz, 12}, {1, 2, -1}},
                          {2, 6, 1});
    array_grid_conf<3, 3>({{cudarrays::compute::xyz, 12}, {2, 1, 0}},
                          {3, 2, 2});
    array_grid_conf<3, 3>({{cudarrays::compute::xyz, 12}, {2, 0, 1}},
                          {3, 2, 2});
    array_grid_conf<3, 3>({{cudarrays::compute::xyz, 12}, {1, 2, 0}},
                          {2, 3, 2});
    array_grid_conf<3, 3>({{cudarrays::compute::xyz, 12}, {1, 0, 2}},
                          {2, 2, 3});
    array_grid_conf<3, 3>({{cudarrays::compute::xyz, 12}, {0, 2, 1}},
                          {2, 3, 2});
    array_grid_conf<3, 3>({{cudarrays::compute::xyz, 12}, {0, 1, 2}},
                          {2, 2, 3});
}

template <unsigned DimsComp, unsigned Dims>
static inline
void array_local_dim_conf(const extents<Dims> &dims,
                          const cudarrays::compute_mapping<DimsComp, Dims> &mapping,
                          const std::array<unsigned, Dims> &result)
{
    auto gpuGrid           = cudarrays::helper_distribution_get_gpu_grid(mapping.comp);
    auto arrayDimToCompDim = mapping.get_array_to_comp(); // TODO: check this function

    auto arrayGrid = cudarrays::helper_distribution_get_array_grid(gpuGrid, arrayDimToCompDim);
    auto localDims = cudarrays::helper_distribution_get_local_dims(dims, arrayGrid);

    for (auto i : utils::make_range(Dims)) {
        if (localDims[i] != result[i]) abort();
        ASSERT_EQ(localDims[i], result[i]);
    }
}

TEST_F(storage_test, local_dims)
{
    extents<1> dims1{120};
    extents<2> dims2{120, 240};
    extents<3> dims3{120, 180, 240};

    //
    // 1D decompositions
    //
    array_local_dim_conf<1, 1>(dims1, {{cudarrays::compute::none, 12}, {-1}},
                               {120});
    array_local_dim_conf<1, 1>(dims1, {{cudarrays::compute::x, 12}, {0}},
                               {10});

    //
    // 2D decompositions
    //
    array_local_dim_conf<2, 2>(dims2, {{cudarrays::compute::none, 12}, {-1, -1}},
                               {120, 240});
    array_local_dim_conf<2, 2>(dims2, {{cudarrays::compute::x, 12}, {-1, 0}},
                               {120, 20});
    array_local_dim_conf<2, 2>(dims2, {{cudarrays::compute::y, 12}, {1, -1}},
                               {10, 240});
    array_local_dim_conf<2, 2>(dims2, {{cudarrays::compute::xy, 12}, {1, 0}},
                               {20, 120});
    array_local_dim_conf<2, 2>(dims2, {{cudarrays::compute::xy, 12}, {0, 1}},
                               {60, 40});

    array_local_dim_conf<2, 3>(dims3, {{cudarrays::compute::xy, 12}, {-1, 1, 0}},
                               {120, 30, 120});
    array_local_dim_conf<2, 3>(dims3, {{cudarrays::compute::xy, 12}, {-1, 0, 1}},
                               {120, 90, 40});

    //
    // 3D decompositions
    //
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::none, 12}, {-1, -1, -1}},
                               {120, 180, 240});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::x, 12}, {-1, -1, 0}},
                               {120, 180, 20});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::y, 12}, {-1, 1, -1}},
                               {120, 15, 240});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::z, 12}, {2, -1, -1}},
                               {10, 180, 240});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::xy, 12}, {-1, 1, 0}},
                               {120, 30, 120});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::xy, 12}, {-1, 0, 1}},
                               {120, 90, 40});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::xz, 12}, {2, -1, 0}},
                               {20, 180, 120});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::xz, 12}, {0, -1, 2}},
                               {60, 180, 40});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::yz, 12}, {2, 1, -1}},
                               {20, 90, 240});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::yz, 12}, {1, 2, -1}},
                               {60, 30, 240});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::xyz, 12}, {2, 1, 0}},
                               {40, 90, 120});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::xyz, 12}, {2, 0, 1}},
                               {40, 90, 120});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::xyz, 12}, {1, 2, 0}},
                               {60, 60, 120});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::xyz, 12}, {1, 0, 2}},
                               {60, 90, 80});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::xyz, 12}, {0, 2, 1}},
                               {60, 60, 120});
    array_local_dim_conf<3, 3>(dims3, {{cudarrays::compute::xyz, 12}, {0, 1, 2}},
                               {60, 90, 80});
}

template <unsigned Dims>
static inline
void array_local_off_conf(const extents<Dims> &dims,
                          const std::array<cudarrays::array_size_t, Dims - 1> &result)
{
    auto localOffs = cudarrays::helper_distribution_get_local_offs(dims);

    for (auto i : utils::make_range(Dims - 1)) {
        ASSERT_EQ(localOffs[i], result[i]);
    }
}

TEST_F(storage_test, array_local_off)
{
    array_local_off_conf<1>({4}, std::array<unsigned, 0u>());

    array_local_off_conf<2>({5, 4}, {4});

    array_local_off_conf<3>({3, 5, 4}, {20, 4});

    array_local_off_conf<4>({7, 3, 5, 4}, {60, 20, 4});
}

template <unsigned Dims>
static inline
void array_local_elems_conf(const extents<Dims> &dims,
                            cudarrays::array_size_t result)
{
    auto elems = cudarrays::helper_distribution_get_local_elems(dims);

    ASSERT_EQ(elems, result);
}

TEST_F(storage_test, array_local_elems)
{
    array_local_elems_conf<1>({4}, 4);

    array_local_elems_conf<2>({5, 4}, 5 * 4);

    array_local_elems_conf<3>({3, 5, 4}, 3 * 5 * 4);

    array_local_elems_conf<4>({7, 3, 5, 4}, 7 * 3 * 5 * 4);
}

template <unsigned Dims>
static inline
void array_gpu_off_conf(const std::array<unsigned, Dims> &grid,
                        const std::array<unsigned, Dims> &result)
{
    auto gpuOffs = cudarrays::helper_distribution_gpu_get_offs(grid);

    for (auto i : utils::make_range(Dims)) {
        ASSERT_EQ(gpuOffs[i], result[i]);
    }
}

TEST_F(storage_test, array_gpu_off)
{
    array_gpu_off_conf<1>({4}, {1});

    array_gpu_off_conf<2>({5, 4}, {4, 1});

    array_gpu_off_conf<3>({3, 5, 4}, {20, 4, 1});

    array_gpu_off_conf<4>({7, 3, 5, 4}, {60, 20, 4, 1});
}

template <size_t DimsComp, size_t Dims>
static inline
void array_dim_to_gpus_conf(const std::array<unsigned, DimsComp> &grid,
                            const std::array<int, Dims>          &arrayToComp,
                            const std::array<unsigned, Dims>     &result)
{
    auto offs = cudarrays::helper_distribution_get_array_dim_to_gpus(grid, arrayToComp);

    for (auto i : utils::make_range(Dims)) {
        ASSERT_EQ(offs[i], result[i]);
    }
}

TEST_F(storage_test, array_dim_to_gpus)
{
    array_dim_to_gpus_conf<1, 1>({4}, {-1}, {0});
    array_dim_to_gpus_conf<1, 1>({4}, {0}, {4});

    array_dim_to_gpus_conf<1, 2>({4}, {0, -1}, {4, 0});
    array_dim_to_gpus_conf<1, 2>({4}, {-1, 0}, {0, 4});

    array_dim_to_gpus_conf<2, 2>({4, 5}, {0, 1}, {4, 5});
    array_dim_to_gpus_conf<2, 2>({4, 5}, {1, 0}, {5, 4});

    array_dim_to_gpus_conf<2, 2>({4, 5}, {-1, 1}, {0, 5});
    array_dim_to_gpus_conf<2, 2>({4, 5}, {-1, 0}, {0, 4});
    array_dim_to_gpus_conf<2, 2>({4, 5}, {0, -1}, {4, 0});
    array_dim_to_gpus_conf<2, 2>({4, 5}, {1, -1}, {5, 0});
}
