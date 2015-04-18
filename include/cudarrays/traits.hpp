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

namespace cudarrays {

template <typename T>
struct array_type_helper {
    using type = T;
};

template <typename T, size_t Elems>
struct array_type_helper<T[Elems]> {
    using type = typename array_type_helper<T>::type;
};

template <typename T, unsigned Dims>
struct array_dim_helper {
    static constexpr unsigned dimensions = Dims;
};

template <typename T, size_t Elems, unsigned Dims>
struct array_dim_helper<T[Elems], Dims> {
    static constexpr unsigned dimensions = array_dim_helper<T, Dims + 1>::dimensions;
};

template <typename T, array_size_t... Extents>
struct array_extents_helper {
    using type = array_extents_helper;

    static constexpr extents<sizeof...(Extents)> get()
    {
        return extents<sizeof...(Extents)>{Extents...};
    };
};

template <typename T, size_t Elems, array_size_t... Extents>
struct array_extents_helper<T[Elems], Extents...> {
    using type = typename array_extents_helper<T, Elems, Extents...>::type;
};

template <typename T, array_size_t... Offsets>
struct array_offsets_helper {
    using type = array_offsets_helper;

    static constexpr extents<sizeof...(Offsets)> get()
    {
        return extents<sizeof...(Offsets)>{Offsets...};
    };
};

template <typename T, size_t Elems, array_size_t Offset, array_size_t... Offsets>
struct array_offsets_helper<T[Elems], Offset, Offsets...> {
    using type = typename array_offsets_helper<T, Elems * Offset, Offset, Offsets...>::type;
};

template <typename T, size_t Elems, array_size_t Offset>
struct array_offsets_helper<T[Elems], Offset> {
    using type = typename array_offsets_helper<T, Elems * Offset, Offset>::type;
};

template <typename T, size_t Elems>
struct array_offsets_helper<T[Elems]> {
    using type = typename array_offsets_helper<T, Elems>::type;
};


template <typename T>
struct array_traits {
    static constexpr unsigned dimensions = array_dim_helper<T, 0>::dimensions;

    using   value_type = typename array_type_helper<T>::type;
    using extents_type = typename array_extents_helper<T>::type;
    using offsets_type = typename array_offsets_helper<T>::type;
};

}

#endif
