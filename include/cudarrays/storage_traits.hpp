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

#pragma once
#ifndef CUDARRAYS_STORAGE_TRAITS_HPP_
#define CUDARRAYS_STORAGE_TRAITS_HPP_

#include "detail/utils/seq.hpp"

#include "common.hpp"

namespace cudarrays {

namespace layout {
struct rmo {};
struct cmo {};

template <unsigned... Order>
struct custom {};
};

namespace detail {

enum class storage_tag {
    AUTO,
    RESHAPE_BLOCK,
    RESHAPE_CYCLIC,
    RESHAPE_BLOCK_CYCLIC,
    VM,
    REPLICATED,
};

static inline std::string
enum_to_string(storage_tag tag)
{
    switch (tag) {
    case storage_tag::AUTO:
        return "AUTO";
    case storage_tag::RESHAPE_BLOCK:
        return "RESHAPE_BLOCK";
    case storage_tag::RESHAPE_CYCLIC:
        return "RESHAPE_CYCLIC";
    case storage_tag::RESHAPE_BLOCK_CYCLIC:
        return "RESHAPE_BLOCK_CYCLIC";
    case storage_tag::VM:
        return "VM";
    case storage_tag::REPLICATED:
        return "REPLICATED";
    default:
        FATAL("Invalid storage_tag value");
    };
}

template <unsigned Dims, typename StorageType>
struct make_dim_order;

template <unsigned Dims>
struct make_dim_order<Dims, cudarrays::layout::rmo> {
    using seq_type = SEQ_GEN_INC(Dims);
};

template <unsigned Dims>
struct make_dim_order<Dims, cudarrays::layout::cmo> {
    using seq_type = SEQ_REVERSE(SEQ_GEN_INC(Dims));
};

template <unsigned Dims, unsigned... Order>
struct make_dim_order<Dims, cudarrays::layout::custom<Order...>> {
    using seq_type = SEQ(Order...);
};
}

template <array_size_t Alignment = 1, array_index_t Offset = 0>
struct align {
    static constexpr array_size_t  alignment = Alignment;
    static constexpr array_index_t offset    = Offset;

    static_assert(Alignment > 0,             "Alignment must be greater than 0");
    static_assert(utils::is_pow2(Alignment), "Alignment must be a power of 2");

    static inline
    constexpr array_size_t get_aligned(const array_size_t &value)
    {
        return alignment == 1 || (value % alignment == 0 && offset == 0)?
            value:
            utils::round_next(value + get_offset(), alignment);
    }

    static inline
    constexpr array_size_t get_offset()
    {
        return alignment == 1?
            0:
            offset > alignment?
                utils::round_next(offset, alignment) - offset:
                offset > 0? alignment - offset: 0;
    }
};

template <array_size_t Alignment, array_index_t Offset>
constexpr array_size_t align<Alignment, Offset>::alignment;
template <array_size_t Alignment, array_index_t Offset>
constexpr array_index_t align<Alignment, Offset>::offset;

using noalign = align<1>;

template <typename T, typename StorageType, typename Align>
struct storage_traits {
using         array_type = T;
    using     alignment_type = Align;
    using  array_traits_type = array_traits<array_type>;
    using         value_type = typename array_traits_type::value_type;

    static constexpr unsigned         dimensions = array_traits_type::dimensions;
    static constexpr unsigned  static_dimensions = array_traits_type::static_dimensions;
    static constexpr unsigned dynamic_dimensions = array_traits_type::dynamic_dimensions;

    // User-provided dimension ordering
    using dim_order_seq = typename detail::make_dim_order<array_traits_type::dimensions, StorageType>::seq_type;
    // Ordered array extents
    using extents_noalign_seq =
        SEQ_REORDER(typename array_traits_type::extents_seq,
                    dim_order_seq);

    static constexpr array_size_t aligned_dim = alignment_type::get_aligned(SEQ_AT(extents_noalign_seq, dimensions - 1));

    // Ordered and aligned array extents
    using extents_align_seq =
        SEQ_SET(extents_noalign_seq,
                dimensions - 1,
                aligned_dim);
    // Nullify static extents if the last physical dimensions are not static
    using extents_seq =
        typename
        std::conditional<SEQ_FIND_LAST(extents_align_seq, 0) == -1 ||
                             utils::is_equal(SEQ_FIND_LAST(extents_align_seq, 0) + 1,
                                             dynamic_dimensions),
                         extents_align_seq,
                         SEQ_GEN_FILL(array_size_t(0), dimensions)
                        >::type;

    // Get offsets for the ordered extents
    using offsets_seq = typename array_offsets_helper<extents_seq>::seq;

    // Type to order elements at run-time
    using permuter_type = utils::permuter<dim_order_seq>;
};

template <typename T, typename StorageType, typename Align>
constexpr unsigned storage_traits<T, StorageType, Align>::dimensions;
template <typename T, typename StorageType, typename Align>
constexpr unsigned storage_traits<T, StorageType, Align>::static_dimensions;
template <typename T, typename StorageType, typename Align>
constexpr unsigned storage_traits<T, StorageType, Align>::dynamic_dimensions;


template <partition Part, unsigned Dims>
struct storage_part_helper {
    using type = typename utils::bitset_to_seq<Part, Dims>::type;
};

template <detail::storage_tag Impl, partition Part>
struct storage_conf {
    static constexpr detail::storage_tag impl = Impl;

    template <unsigned Dims>
    using part_seq = typename storage_part_helper<Part, Dims>::type;
};


template <typename T, typename StorageType, typename Align, typename PartConf>
struct dist_storage_traits :
    storage_traits<T, StorageType, Align> {
    using parent_type = storage_traits<T, StorageType, Align>;

    // Order dimension partitioning configuration
    using partitioning_seq =
        SEQ_REORDER(
            typename PartConf::template part_seq<parent_type::array_traits_type::dimensions>,
            typename parent_type::dim_order_seq);
    static constexpr partition partition_value =
        partition(utils::seq_to_bitset<partitioning_seq>::value);
};

template <typename T, typename StorageType, typename Align, typename PartConf>
constexpr partition dist_storage_traits<T, StorageType, Align, PartConf>::partition_value;

}

#endif
