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
#ifndef CUDARRAYS_DETAIL_UTILS_BITSET_HPP_
#define CUDARRAYS_DETAIL_UTILS_BITSET_HPP_

#include "seq.hpp"

namespace utils {

namespace detail __attribute__ ((visibility ("hidden"))) {
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
    constexpr static unsigned value =
        unsigned(Val) << (Idx - 1) | seq_to_bitset<Idx - 1, SEQ_WITH_TYPE(bool, Vals...)>::value;
};

template <>
struct seq_to_bitset<0, SEQ_WITH_TYPE(bool)> {
    constexpr static unsigned value = 0;
};
}

template <unsigned Bits, unsigned N>
struct bitset_to_seq {
    using type = typename detail::bitset_to_seq<Bits, SEQ_GEN_INC(N)>::type;
};

template <typename S>
struct seq_to_bitset {
    constexpr static unsigned value = detail::seq_to_bitset<SEQ_SIZE(S), S>::value;
};

template <typename S>
constexpr unsigned seq_to_bitset<S>::value;


// TODO: make this constexpr in C++14
template <unsigned N>
inline static
std::array<bool, N>
bitset_to_array(unsigned b)
{
    std::array<bool, N> ret = {{}};
    for (int i = N - 1; i >= 0; --i)
        ret[(N - 1) - i] = (b & (1 << i)) != 0;

    return ret;
}

template <unsigned N>
inline static
unsigned
array_to_bitset(const std::array<bool, N> &b)
{
    unsigned ret = 0;
    for (unsigned i = 0; i < N; ++i)
        ret |= (unsigned(b[i]) << ((N - 2) - i));

    return ret;
}

}

#endif // CUDARRAYS_DETAIL_UTILS_BITSET_HPP_
