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
#ifndef CUDARRAYS_DETAIL_UTILS_PERMUTE_HPP_
#define CUDARRAYS_DETAIL_UTILS_PERMUTE_HPP_

#include <array>

#include "../../compiler.hpp"

#include "log.hpp"
#include "seq.hpp"
#include "stl.hpp"

namespace utils {

template <typename C, typename M>
static C reorder_scatter(const C &cont, const M &map)
{
    ASSERT(cont.size() == map.size());
    C ret;
    for (unsigned i = 0; i < cont.size(); ++i) {
        unsigned j = map[i];
        ret[j] = cont[i];
    }
    return ret;
}

template <typename C, typename M>
static C reorder_gather(const C &cont, const M &map)
{
    ASSERT(cont.size() == map.size());
    C ret;
    for (unsigned i = 0; i < cont.size(); ++i) {
        unsigned j = map[i];
        ret[i] = cont[j];
    }
    return ret;
}

template <typename T, size_t Elems>
static std::array<T, Elems>
make_array(const T(&array)[Elems])
{
    std::array<T, Elems> ret;

    copy(array, ret);

    return ret;
}

template <unsigned N, class M>
struct permuter_detail;

template <unsigned Idx0>
struct permuter_detail<1, SEQ_WITH_TYPE(unsigned, Idx0)> {
    template <unsigned IdxSelect, typename IdxType>
    static inline __hostdevice__
    IdxType select(IdxType idx)
    {
        static_assert(IdxSelect < 1, "Invalid index");

        return idx;
    }

    template <typename C>
    static inline __hostdevice__
    C reorder(const C &cont)
    {
        return cont;
    }

    template <unsigned Orig>
    static inline __host__ __device__
    unsigned dim_index()
    {
        return 0;
    }

    static inline __host__ __device__
    unsigned dim_index(unsigned)
    {
        return 0;
    }

};

template <unsigned Idx0, unsigned Idx1>
struct permuter_detail<2, SEQ_WITH_TYPE(unsigned, Idx0, Idx1)> {
    static inline __host__
    std::array<unsigned, 2> as_array()
    {
        return {Idx0, Idx1};
    }

    template <unsigned IdxSelect, typename IdxType>
    static inline __hostdevice__
    IdxType select(IdxType idx0, IdxType idx1)
    {
        static_assert(IdxSelect < 2, "Invalid index");

        if (IdxSelect == Idx0)
            return idx0;
        else // if (IdxSelect == Idx1)
            return idx1;
    }

    template <typename C>
    static inline __host__
    C reorder(const C &cont)
    {
        return reorder_gather(cont, as_array());
    }

    template <unsigned Orig>
    static inline __host__ __device__
    unsigned dim_index()
    {
        if (Orig == Idx0)
            return 0;
        else // if (Orig == Idx1)
            return 1;
    }

    static inline __host__ __device__
    unsigned dim_index(unsigned orig)
    {
        if (orig == Idx0)
            return 0;
        else // if (orig == Idx1)
            return 1;
    }
};

template <unsigned Idx0, unsigned Idx1, unsigned Idx2>
struct permuter_detail<3, SEQ_WITH_TYPE(unsigned, Idx0, Idx1, Idx2)> {
    static inline __host__
    std::array<unsigned, 3> as_array()
    {
        return {Idx0, Idx1, Idx2};
    }

    template <unsigned IdxSelect, typename IdxType>
    static inline __hostdevice__
    IdxType select(IdxType idx0, IdxType idx1, IdxType idx2)
    {
        static_assert(IdxSelect < 3, "Invalid index");

        if (IdxSelect == Idx0)
            return idx0;
        else if (IdxSelect == Idx1)
            return idx1;
        else // if (IdxSelect == Idx2)
            return idx2;
    }

    template <typename C>
    static inline __host__
    C reorder(const C &cont)
    {
        return reorder_gather(cont, as_array());
    }

    template <unsigned Orig>
    static inline __host__ __device__
    unsigned dim_index()
    {
        if (Orig == Idx0)
            return 0;
        else if (Orig == Idx1)
            return 1;
        else // if (IdxSelect == Idx2)
            return 2;
    }

    static inline __host__ __device__
    unsigned dim_index(unsigned orig)
    {
        if (orig == Idx0)
            return 0;
        else if (orig == Idx1)
            return 1;
        else // if (IdxSelect == Idx2)
            return 2;
    }
};

template <typename M>
using permuter = permuter_detail<SEQ_SIZE(M), M>;

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
