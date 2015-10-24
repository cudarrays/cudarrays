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
#ifndef CUDARRAYS_TRAITS_HPP_
#define CUDARRAYS_TRAITS_HPP_

#include <cstddef>

#include "common.hpp"
#include "utils.hpp"

namespace cudarrays {

template <typename T>
struct array_type_helper {
    using type = T;
};

template <typename T, size_t Elems>
struct array_type_helper<T[Elems]> {
    static_assert(Elems != 0, "Extents for a static dimension cannot be zero.");
    using type = typename array_type_helper<T>::type;
};

template <typename T>
struct array_type_helper<T *> {
    using type = typename array_type_helper<T>::type;
};

namespace detail __attribute__ ((visibility ("hidden"))) {

template <typename T, array_size_t... Extents>
struct array_extents_helper {
    using seq = SEQ(Extents...);
};

template <typename T, size_t Elems, array_size_t... Extents>
struct array_extents_helper<T[Elems], Extents...> {
    static_assert(Elems != 0, "Bounds of a static dimension cannot be zero.");
    // Array elements are interpreted right-to-left, pushing dimension size at the end
    using seq = typename array_extents_helper<T, Extents..., Elems>::seq;
};

template <typename T, array_size_t... Extents>
struct array_extents_helper<T *, Extents...> {
    // Pointers are interpreted left-to-right, pushing dimension size at the beginning
    using seq = typename array_extents_helper<T, 0, Extents...>::seq;
};

}

template <typename T>
struct array_extents_helper {
    using seq = typename detail::array_extents_helper<T>::seq;

    static constexpr unsigned         dimensions = SEQ_SIZE(seq);
    static constexpr unsigned dynamic_dimensions = SEQ_COUNT(seq, 0);
    static constexpr unsigned  static_dimensions = dimensions - dynamic_dimensions;
};

// Compile-time array offset generation. It generates N - 1 offsets
namespace detail {

template <typename S, unsigned Dim, array_size_t... Offset>
struct array_offsets_helper;

template <typename S, array_size_t Offset, array_size_t... Offsets>
struct array_offsets_helper<S, 0u, Offset, Offsets...> {
    using seq = SEQ_WITH_TYPE(array_size_t, Offsets...);
};

template <typename S, unsigned Dim, array_size_t Offset, array_size_t... Offsets>
struct array_offsets_helper<S, Dim, Offset, Offsets...> {
    using seq = typename array_offsets_helper<S,
                                              Dim - 1,
                                              Offset * SEQ_AT(S, Dim - 1),
                                              Offset,
                                              Offsets...>::seq;
};

}

template <typename S>
using array_offsets_helper = detail::array_offsets_helper<S, SEQ_SIZE(S) - 1, SEQ_AT(S, SEQ_SIZE(S) - 1)>;

template <typename T>
struct array_traits {
    using  value_type = typename    array_type_helper<T>::type;
    using extents_seq = typename array_extents_helper<T>::seq;
    using offsets_seq = typename array_offsets_helper<extents_seq>::seq;

    static constexpr unsigned         dimensions = SEQ_SIZE(extents_seq);
    static constexpr unsigned dynamic_dimensions = SEQ_COUNT(extents_seq, 0);
    static constexpr unsigned  static_dimensions = dimensions - dynamic_dimensions;

    static extents<dimensions>
    make_extents(const extents<dynamic_dimensions> &ext)
    {
        extents<dimensions> ret = extents_seq::as_array();

        unsigned idx = 0;
        for (auto &e: ret) {
            if (e == 0)
                e = ext[idx++];
        }

        return ret;
    }
};

template <typename T>
constexpr unsigned array_traits<T>::dimensions;
template <typename T>
constexpr unsigned array_traits<T>::static_dimensions;
template <typename T>
constexpr unsigned array_traits<T>::dynamic_dimensions;

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
