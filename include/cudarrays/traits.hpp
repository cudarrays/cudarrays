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

#ifndef CUDARRAYS_TRAITS_HPP_
#define CUDARRAYS_TRAITS_HPP_

#include <cstddef>

#include "common.hpp"
#include "config.hpp"
#include "utils.hpp"

namespace cudarrays {

template <typename T>
struct array_type_helper {
    using type = T;
};

template <typename T, size_t Elems>
struct array_type_helper<T[Elems]> {
    static_assert(Elems != 0, "Bounds of a static dimension cannot be zero.");
    using type = typename array_type_helper<T>::type;
};

template <typename T>
struct array_type_helper<T *> {
    using type = typename array_type_helper<T>::type;
};

template <typename T, unsigned StaticDims, unsigned DynamicDims, array_size_t... Extents>
struct array_extents_helper_internal {
    using type = array_extents_helper_internal;
    static constexpr unsigned dimensions         = StaticDims + DynamicDims;

    static constexpr unsigned static_dimensions  = StaticDims;
    static constexpr unsigned dynamic_dimensions = DynamicDims;

    template <unsigned Idx>
    static constexpr array_size_t get()
    {
        return utils::mpl::seq_at<Idx, array_size_t, Extents...>::value;
    }

    static extents<dimensions> get(const extents<dynamic_dimensions> &ext)
    {
        extents<dimensions> ret{Extents...};

        unsigned idx = 0;
        for (auto &e: ret) {
            if (e == 0)
                e = ext[idx++];
        }

        return ret;
    }
};

template <typename T, unsigned StaticDims, unsigned DynamicDims, array_size_t... Extents>
constexpr unsigned array_extents_helper_internal<T, StaticDims, DynamicDims, Extents...>::dimensions;
template <typename T, unsigned StaticDims, unsigned DynamicDims, array_size_t... Extents>
constexpr unsigned array_extents_helper_internal<T, StaticDims, DynamicDims, Extents...>::static_dimensions;
template <typename T, unsigned StaticDims, unsigned DynamicDims, array_size_t... Extents>
constexpr unsigned array_extents_helper_internal<T, StaticDims, DynamicDims, Extents...>::dynamic_dimensions;

template <typename T, unsigned StaticDims, unsigned DynamicDims, size_t Elems, array_size_t... Extents>
struct array_extents_helper_internal<T[Elems], StaticDims, DynamicDims, Extents...> {
    static_assert(Elems != 0, "Bounds of a static dimension cannot be zero.");

    // Array elements are interpreted differently, pushing dimension size at the end
    using type = typename array_extents_helper_internal<T, StaticDims + 1, DynamicDims, Extents..., Elems>::type;
};

template <typename T, unsigned StaticDims, unsigned DynamicDims, array_size_t... Extents>
struct array_extents_helper_internal<T *, StaticDims, DynamicDims, Extents...> {
    using type = typename array_extents_helper_internal<T, StaticDims, DynamicDims + 1, 0, Extents...>::type;
};

template <typename T>
struct array_extents_helper {
    using type = typename array_extents_helper_internal<T, 0, 0>::type;
};

template <typename T, T A, T B>
struct multiply {
    static constexpr T value = A * B;
};

template <typename T, array_size_t... Offset>
struct array_offsets_helper;

template <typename T, array_size_t Offset, array_size_t... Offsets>
struct array_offsets_helper<T, Offset, Offsets...> {
    using type = array_offsets_helper;

    template <size_t Idx>
    static constexpr array_size_t get()
    {
        return utils::mpl::seq_at<Idx, array_size_t, Offsets...>::value;
    }
};

template <typename T, size_t Elems, array_size_t Offset, array_size_t... Offsets>
struct array_offsets_helper<T[Elems], Offset, Offsets...> {
    static_assert(Elems != 0, "Bounds of a static dimension cannot be zero.");

    using type = typename array_offsets_helper<T,
                                              multiply<array_size_t, Offsets, Elems>::value ...,
                                              multiply<array_size_t, Offset, Elems>::value,
                                              Elems>::type;
};

template <typename T,  array_size_t Offset, array_size_t... Offsets>
struct array_offsets_helper<T *, Offset, Offsets...> {
    using type = typename array_offsets_helper<T, 0, Offset, Offsets...>::type;
};

template <typename T, size_t Elems>
struct array_offsets_helper<T[Elems]> {
    static_assert(Elems != 0, "Bounds of a static dimension cannot be zero.");

    using type = typename array_offsets_helper<T, Elems>::type;
};

template <typename T>
struct array_offsets_helper<T *> {
    using type = typename array_offsets_helper<T, 0>::type;
};

template <typename T>
struct array_traits {
    static constexpr unsigned         dimensions = array_extents_helper<T>::type::dimensions;
    static constexpr unsigned  static_dimensions = array_extents_helper<T>::type::static_dimensions;
    static constexpr unsigned dynamic_dimensions = array_extents_helper<T>::type::dynamic_dimensions;

    using   value_type = typename array_type_helper<T>::type;
    using extents_type = typename array_extents_helper<T>::type;
    using offsets_type = typename array_offsets_helper<T>::type;
};

}

#endif
