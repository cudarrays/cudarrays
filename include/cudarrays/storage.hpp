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
#ifndef CUDARRAYS_STORAGE_HPP_
#define CUDARRAYS_STORAGE_HPP_

#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <functional>
#include <type_traits>

#include "common.hpp"
#include "traits.hpp"
#include "utils.hpp"

namespace cudarrays {

namespace layout {
struct rmo {};
struct cmo {};

template <unsigned... Order>
struct custom {};
};

enum class storage_tag {
    AUTO,
    RESHAPE_BLOCK,
    RESHAPE_CYCLIC,
    RESHAPE_BLOCK_CYCLIC,
    VM,
    REPLICATED,
};

static std::string
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

template <partition Part, unsigned Dims>
struct storage_part_dim_helper {
    static constexpr size_t dimensions = Dims;

    static_assert(dimensions <= 3,
                  "Up to 3 dimensional arrays are supported so far");

    static constexpr bool X = bool(partition::X & Part);
    static constexpr bool Y = bool(partition::Y & Part);
    static constexpr bool Z = bool(partition::Z & Part);
};


namespace detail {
template <unsigned Bits, typename Idxs>
struct bitset_to_seq;

template <unsigned Bits, unsigned ...Idxs>
struct bitset_to_seq<Bits, SEQ_WITH_TYPE(unsigned, Idxs...)> {
    using type = SEQ_REVERSE(SEQ_WITH_TYPE(bool, (0u != (Bits & (1 << Idxs)))...));
};

template <unsigned Idx, typename Seq>
struct seq_to_bitset;

template <unsigned Idx, bool Val, bool ...Vals>
struct seq_to_bitset<Idx, SEQ_WITH_TYPE(bool, Val, Vals...)> {
    constexpr static unsigned value()
    {
        return unsigned(Val) << (Idx - 1) | seq_to_bitset<Idx - 1, SEQ_WITH_TYPE(bool, Vals...)>::value();
    }
};

template <>
struct seq_to_bitset<0, SEQ_WITH_TYPE(bool)> {
    constexpr static unsigned value()
    {
        return 0;
    }
};
}

template <unsigned Bits, unsigned N>
struct bitset_to_seq {
    using type = typename detail::bitset_to_seq<Bits, SEQ_GEN_INC(N)>::type;
};

template <typename S>
struct seq_to_bitset {
    constexpr static unsigned value()
    {
        return detail::seq_to_bitset<SEQ_SIZE(S), S>::value();
    }
};

template <partition Part, unsigned Dims>
struct storage_part_helper {
    using type = typename bitset_to_seq<Part, Dims>::type;
};

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

template <storage_tag Storage>
struct select_auto_impl {
    static constexpr storage_tag impl = Storage;
};

template <>
struct select_auto_impl<storage_tag::AUTO> {
    static constexpr storage_tag impl = storage_tag::REPLICATED;
};

template <storage_tag Storage, partition Part>
struct storage_conf {
    static constexpr storage_tag       impl = Storage;
    static constexpr storage_tag final_impl = select_auto_impl<Storage>::impl;
    template <unsigned Dims>
    using part_seq = typename storage_part_helper<Part, Dims>::type;
};

template <storage_tag Impl>
struct storage_part
{
    static constexpr storage_tag       impl = Impl;
    static constexpr storage_tag final_impl = select_auto_impl<Impl>::impl;

    using none = storage_conf<Impl, partition::NONE>;

    using x = storage_conf<Impl, partition::X>;
    using y = storage_conf<Impl, partition::Y>;
    using z = storage_conf<Impl, partition::Z>;

    using xy = storage_conf<Impl, partition::XY>;
    using xz = storage_conf<Impl, partition::XZ>;
    using yz = storage_conf<Impl, partition::YZ>;

    using xyz = storage_conf<Impl, partition::XYZ>;

    static std::string name()
    {
        return enum_to_string(Impl);
    }
};

template <typename T, T Val1, T Val2>
struct is_greater {
    static constexpr bool value = Val1 > Val2;
};

template <typename T, T Val1, T Val2>
struct is_less {
    static constexpr bool value = Val1 < Val2;
};

template <typename T, T Val1, T Val2>
struct is_equal {
    static constexpr bool value = (Val1 == Val2);
};

template <typename T, typename StorageType, typename PartConf>
struct storage_traits {
    using         array_type = T;
    using  array_traits_type = array_traits<array_type>;

    static constexpr unsigned         dimensions = array_traits_type::dimensions;
    static constexpr unsigned  static_dimensions = array_traits_type::static_dimensions;
    static constexpr unsigned dynamic_dimensions = array_traits_type::dynamic_dimensions;

    // User-provided dimension ordering
    using dim_order_seq = typename make_dim_order<array_traits_type::dimensions, StorageType>::seq_type;
    // Order array extents
    using extents_pre_seq =
        SEQ_REORDER(
            typename array_traits_type::extents_seq,
            dim_order_seq);
    // Nullify static extents if the last physical dimensions are not static
    using extents_seq =
        typename
        std::conditional<SEQ_FIND_LAST(extents_pre_seq, 0) == -1 ||
                             is_equal<ssize_t,
                                      SEQ_FIND_LAST(extents_pre_seq, 0) + 1,
                                      dynamic_dimensions
                                     >::value,
                         extents_pre_seq,
                         SEQ_GEN_FILL(array_size_t(0), dimensions)
                        >::type;

    // Get offsets for the ordered extents
    using offsets_seq = typename array_offsets_helper<extents_seq>::seq;

    // Order dimension partitioning configuration
    using partitioning_seq =
        SEQ_REORDER(
            typename PartConf::template part_seq<array_traits_type::dimensions>,
            dim_order_seq);
    static constexpr partition partition_value =
        partition(seq_to_bitset<partitioning_seq>::value());

    // Type to order elements at run-time
    using permuter_type = utils::permuter<dim_order_seq>;
};

template <typename T, typename StorageType, typename PartConf>
constexpr unsigned storage_traits<T, StorageType, PartConf>::dimensions;
template <typename T, typename StorageType, typename PartConf>
constexpr unsigned storage_traits<T, StorageType, PartConf>::static_dimensions;
template <typename T, typename StorageType, typename PartConf>
constexpr unsigned storage_traits<T, StorageType, PartConf>::dynamic_dimensions;
template <typename T, typename StorageType, typename PartConf>
constexpr partition storage_traits<T, StorageType, PartConf>::partition_value;

using automatic            = storage_part<storage_tag::AUTO>;
using reshape_block        = storage_part<storage_tag::RESHAPE_BLOCK>;
using reshape_cyclic       = storage_part<storage_tag::RESHAPE_CYCLIC>;
using reshape_block_cyclic = storage_part<storage_tag::RESHAPE_BLOCK_CYCLIC>;
using vm                   = storage_part<storage_tag::VM>;
using replicate            = storage_part<storage_tag::REPLICATED>;

using reshape = reshape_block;

}

#endif
/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
