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
#include "array_traits.hpp"
#include "storage_traits.hpp"
#include "utils.hpp"

namespace cudarrays {

template <partition Part, size_t Dims>
struct storage_part_dim_helper {
    static constexpr size_t dimensions = Dims;

    static_assert(dimensions <= 3,
                  "Up to 3 dimensional arrays are supported so far");

    static constexpr bool X = bool(partition::X & Part);
    static constexpr bool Y = bool(partition::Y & Part);
    static constexpr bool Z = bool(partition::Z & Part);
};

template <detail::storage_tag Impl>
struct storage_part
{
    static constexpr detail::storage_tag impl = Impl;

    using none = storage_conf<impl, partition::NONE>;

    using   x = storage_conf<impl, partition::X>;
    using   y = storage_conf<impl, partition::Y>;
    using   z = storage_conf<impl, partition::Z>;

    using  xy = storage_conf<impl, partition::XY>;
    using  xz = storage_conf<impl, partition::XZ>;
    using  yz = storage_conf<impl, partition::YZ>;

    using xyz = storage_conf<impl, partition::XYZ>;

    static std::string name()
    {
        return detail::enum_to_string(impl);
    }
};

namespace detail {
static constexpr
detail::storage_tag
select_auto_impl(storage_tag given)
{
    return given == storage_tag::AUTO? storage_tag::REPLICATED:
                                       given;
}
}

using automatic            = storage_part<detail::select_auto_impl(detail::storage_tag::AUTO)>;
using reshape_block        = storage_part<detail::storage_tag::RESHAPE_BLOCK>;
using reshape_cyclic       = storage_part<detail::storage_tag::RESHAPE_CYCLIC>;
using reshape_block_cyclic = storage_part<detail::storage_tag::RESHAPE_BLOCK_CYCLIC>;
using vm                   = storage_part<detail::storage_tag::VM>;
using replicate            = storage_part<detail::storage_tag::REPLICATED>;

using reshape = reshape_block;

}

#endif
/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
