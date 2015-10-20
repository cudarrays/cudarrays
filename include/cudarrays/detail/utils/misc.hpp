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
 * to use, copy, modify, seq_merge, publish, distribute, sublicense, and/or sell
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

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <string>
#include <vector>

#include "../../common.hpp"

namespace utils {

template <typename T>
void
_get_factors(T n, std::vector<T> &factors)
{
    if (n <= 1) return;

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
    T res = a / b;
    if (a % b != 0) ++res;
    return res;
}

template <class T, class U>
constexpr typename std::common_type<T, U>::type
round_next(const T &val, const U &step)
{
    static_assert(std::is_integral<T>::value &&
                  std::is_integral<U>::value, "round_next works on integral types only");
    return (val % step != 0)? ((val / step) + 1) * step:
                              val;
}

template <typename T>
static inline
constexpr bool is_pow2(T val)
{
    return ((val & 1) != 0) && val > 1?
            false:
            (val == 1? true:
                       is_pow2(val >> 1));
}

template <typename T, typename U>
static inline
constexpr bool is_greater(T val1, U val2)
{
    return val1 > val2;
}

template <typename T, typename U>
static inline
constexpr bool is_less(T val1, U val2)
{
    return val1 < val2;
}

template <typename T, typename U>
static inline
constexpr bool is_equal(T val1, U val2)
{
    return val1 == val2;
}

#ifdef CUDARRAYS_UNITTEST
#define CUDARRAYS_TESTED(C,T) friend C##_##T##_Test;
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

static inline std::string
string_replace_all(std::string str, const std::string& from, const std::string& to)
{
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}

static inline std::vector<std::string>
string_tokenize(std::string str, const std::string& delimiter)
{
    std::vector<std::string> ret;
    // Skip delimiters at beginning.
    auto lastPos = str.find_first_not_of(delimiter, 0);
    // Find first "non-delimiter".
    auto pos     = str.find_first_of(delimiter, lastPos);

    while (pos != std::string::npos || lastPos != std::string::npos) {
        // Found a token, add it to the vector.
        ret.push_back(str.substr(lastPos, pos - lastPos));
        // Skip delimiters.  Note the "not_of"
        lastPos = str.find_first_not_of(delimiter, pos);
        // Find next "non-delimiter"
        pos = str.find_first_of(delimiter, lastPos);
    }

    return ret;
}

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
