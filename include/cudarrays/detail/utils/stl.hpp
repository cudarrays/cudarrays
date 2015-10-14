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
#ifndef CUDARRAYS_DETAIL_UTILS_STL_HPP_
#define CUDARRAYS_DETAIL_UTILS_STL_HPP_

#include <algorithm>
#include <array>
#include <numeric>
#include <sstream>
#include <utility>

namespace std {

template <typename T>
static T *
begin(T(&)[0])
{
    return nullptr;
}

template <typename T>
static T *
end(T(&)[0])
{
    return nullptr;
}

}

namespace utils {

template <typename C, typename T, typename BinaryOperation>
static inline
auto
accumulate(const C &cont, T &&init, BinaryOperation &&op) -> decltype(std::accumulate(std::begin(cont), std::end(cont), init, op))
{
    return std::accumulate(std::begin(cont), std::end(cont), std::forward<T>(init), std::forward<BinaryOperation>(op));
}

template <typename C1, typename C2>
static inline
auto
copy(const C1 &in, C2 &out) -> decltype(std::copy(std::begin(in), std::end(in), std::begin(out)))
{
    return std::copy(std::begin(in), std::end(in), std::begin(out));
}

template <typename C, typename T>
static inline
auto
count(const C &cont, T &&val) -> decltype(std::count(std::begin(cont), std::end(cont), val))
{
    return std::count(std::begin(cont), std::end(cont), std::forward<T>(val));
}

template <typename C, typename P>
static inline
auto
count_if(const C &cont, P &&pred) -> decltype(std::count_if(std::begin(cont), std::end(cont), pred))
{
    return std::count_if(std::begin(cont), std::end(cont), std::forward<P>(pred));
}


template <typename C>
static inline
auto
equal(const C &a, const C &b) -> decltype(std::equal(std::begin(a), std::end(a), std::begin(b)))
{
    return std::equal(std::begin(a), std::end(a), std::begin(b));
}

template <typename C, typename T>
static inline
auto
fill(C &cont, T &&val) -> decltype(std::fill(std::begin(cont), std::end(cont), val))
{
    return std::fill(std::begin(cont), std::end(cont), std::forward<T>(val));
}

template <typename C>
static inline
auto
sort(C &cont) -> decltype(std::sort(std::begin(cont), std::end(cont)))
{
    return std::sort(std::begin(cont), std::end(cont));
}

template <typename C, typename Compare>
static inline
auto
sort(C &cont, Compare &&op) -> decltype(std::sort(std::begin(cont), std::end(cont), std::forward<Compare>(op)))
{
    return std::sort(std::begin(cont), std::end(cont), std::forward<Compare>(op));
}

template <typename T>
std::string
to_string(const T &obj)
{
    std::stringstream str_tmp;

    size_t elems = obj.size();
    size_t i = 0;
    for (auto &val : obj) {
        str_tmp << val;
        if (++i < elems) str_tmp << ", ";
    }

    return str_tmp.str();
}

template <bool B, class T = void>
using enable_if_t = typename std::enable_if<B,T>::type;

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
