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
#ifndef CUDARRAYS_DETAIL_UTILS_MISC_HPP_
#define CUDARRAYS_DETAIL_UTILS_MISC_HPP_

#include <type_traits>
#include <vector>

namespace utils {

template <typename T, class V1, class V2>
struct merge_static;

template <typename T, T... Values1, T... Values2, template <T...> class S>
struct merge_static<T, S<Values1...>, S<Values2...>> {
    using type = S<Values1..., Values2...>;
};

template <typename T, unsigned Size, T FillValue, template <T...> class S, T... Generated>
struct generate_static_detail {
    using type = typename generate_static_detail<T, Size - 1, FillValue, S, FillValue, Generated...>::type;
};

template <typename T, T FillValue, template <T...> class S, T... Generated>
struct generate_static_detail<T, 0, FillValue, S, Generated...> {
    using type = S<Generated...>;
};

template <typename T, T FillValue, template <T...> class S>
struct generate_static_detail<T, 0, FillValue, S> {
    using type = S<>;
};

template <typename T, unsigned Size, T FillValue, template <T...> class S>
struct generate_static {
    using type = generate_static_detail<T, Size - 1, FillValue, S, FillValue>;
};

template <typename T, T FillValue, template <T...> class S>
struct generate_static<T, 0, FillValue, S> {
    using type = generate_static_detail<T, 0, FillValue, S>;
};

template <typename T, unsigned Size, T FillValue, template <T...> class S, T... Rest>
struct prepend_static {
    using prepend_values_type = generate_static<T, Size - sizeof...(Rest), FillValue, S>;
    using type = typename merge_static<T, prepend_values_type, S<Rest...>>::type;
};

template <typename T, unsigned Size, T FillValue, template <T...> class S, T... Rest>
struct append_static {
    using append_values_type = generate_static<T, Size - sizeof...(Rest), FillValue, S>;
    using type = typename merge_static<T, S<Rest...>, append_values_type>::type;
};

template <typename T>
void
_get_factors(T n, std::vector<T> &factors)
{
    if (n == 1) return;

    unsigned z = 2;
    while (z <= n) {
        if (n % z == 0) {
            break;
        }
        ++z;
    }
    factors.push_back(z);
    _get_factors(n/z, factors);
    return;
}

template <typename T>
std::vector<T>
get_factors(T n)
{
    std::vector<T> ret;

    _get_factors(n, ret);

    return ret;
}

template <class T, class U>
T div_ceil(T a, U b)
{
    static_assert(std::is_integral<T>::value &&
                  std::is_integral<U>::value, "div_ceil works on integral types only");
    ASSERT(b > U(0));
    T res = a / b;
    if (a % b != 0) ++res;
    return res;
}

template <class T, class U>
typename std::common_type<T, U>::type
round_next(T val, U step)
{
    static_assert(std::is_integral<T>::value &&
                  std::is_integral<U>::value, "round_next works on integral types only");
    if (val % step != 0)
        val = ((val / step) + 1) * step;
    return val;
}

#ifdef CUDARRAYS_UNITTEST
#define CUDARRAYS_TESTED(C,T) friend ::C##_##T##_Test;
#else
#define CUDARRAYS_TESTED(C,T)
#endif

#define CUDA_CALL(x)                                       \
do {                                                       \
    cudaError_t err__ = (x);                               \
    if (err__ != cudaSuccess) {                            \
        fprintf(stderr,                                    \
                "Error calling CUDA: %d. Message: '%s'\n", \
                err__,                                     \
                cudaGetErrorString(err__));                \
        abort();                                           \
    }                                                      \
} while (0)

#ifdef CUDARRAYS_UNITTEST
#define CUDARRAYS_TESTED(C,T) friend ::C##_##T##_Test;
#else
#define CUDARRAYS_TESTED(C,T)
#endif

#define CUDA_CALL(x)                                       \
do {                                                       \
    cudaError_t err__ = (x);                               \
    if (err__ != cudaSuccess) {                            \
        fprintf(stderr,                                    \
                "Error calling CUDA: %d. Message: '%s'\n", \
                err__,                                     \
                cudaGetErrorString(err__));                \
        abort();                                           \
    }                                                      \
} while (0)

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
