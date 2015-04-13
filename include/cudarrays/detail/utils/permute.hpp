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

// TODO: Life will be much more easier when NVCC compiles this
#if 0
template <typename T, class U, class M>
struct reorder_gather_static;

template <typename T, T... Vals,
          template <T...> class U,
          unsigned... Idxs,
          template <unsigned...> class M>
struct reorder_gather_static<T, U<Vals...>, M<Idxs...>> {
    static_assert(sizeof...(Vals) == sizeof...(Idxs),
                  "Wrong reorder configuration");
    static constexpr array_dev<T, sizeof...(Vals)> my_vals{Vals...};
    static constexpr array_dev<unsigned, sizeof...(Idxs)> my_idxs{Idxs...};

    using type = U<my_vals[Idxs]...>;
};
#else

template <typename T, size_t Elems>
struct array_dev {
    T data_[Elems];

    __hostdevice__
    constexpr
    T operator[](size_t idx) const
    {
        return data_[idx];
    }

    __hostdevice__
    constexpr
    const T &at(size_t idx) const
    {
        return data_[idx];
    }

    template <size_t Idx>
    __hostdevice__
    constexpr
    const typename std::enable_if<(Idx < Elems), T>::type at() const
    {
        return data_[Idx];
    }

    template <size_t Idx>
    __hostdevice__
    constexpr
    const typename std::enable_if<(Idx >= Elems), T>::type at() const
    {
        return T(0);
    }

    __hostdevice__
    constexpr
    size_t size() const
    {
        return Elems;
    }
};

template <unsigned Idx, typename T, T...>
struct selector;

template <typename T, T Val0>
struct selector<0, T, Val0> {
    static constexpr T value = Val0;
};

template <typename T, T Val0, T... ValsPost>
struct selector<0, T, Val0, ValsPost...> {
    static constexpr T value = Val0;
};

template <typename T, T Val0, T Val1>
struct selector<1, T, Val0, Val1> {
    static constexpr T value = Val1;
};

template <typename T, T Val0, T Val1, T... Vals>
struct selector<1, T, Val0, Val1, Vals...> {
    static constexpr T value = Val1;
};

template <typename T, T Val0, T Val1, T Val2>
struct selector<2, T, Val0, Val1, Val2> {
    static constexpr T value = Val2;
};

template <typename T, T Val0, T Val1, T Val2, T... Vals>
struct selector<2, T, Val0, Val1, Val2, Vals...> {
    static constexpr T value = Val2;
};

template <unsigned N, typename T, class U, class M>
struct reorder_gather_static;

template <typename T, T... Vals,
          template <T...> class U,
          unsigned Idx0,
          template <unsigned...> class M>
struct reorder_gather_static<1, T, U<Vals...>, M<Idx0>> {
    static_assert(sizeof...(Vals) == 1,
                  "Wrong reorder configuration");
    using order_type = M<Idx0>;
    using type = U<Vals...>;
};

template <typename T, T... Vals,
          template <T...> class U,
          unsigned Idx0, unsigned Idx1,
          template <unsigned...> class M>
struct reorder_gather_static<2, T, U<Vals...>, M<Idx0, Idx1>> {
    static_assert(sizeof...(Vals) == 2,
                  "Wrong reorder configuration");
    using order_type = M<Idx0, Idx1>;
    using type = U<selector<Idx0, T, Vals...>::value,
                   selector<Idx1, T, Vals...>::value>;
};

template <typename T, T... Vals,
          template <T...> class U,
          unsigned Idx0, unsigned Idx1, unsigned Idx2,
          template <unsigned...> class M>
struct reorder_gather_static<3, T, U<Vals...>, M<Idx0, Idx1, Idx2>> {
    static_assert(sizeof...(Vals) == 3,
                  "Wrong reorder configuration");
    using order_type = M<Idx0, Idx1, Idx2>;
    using type = U<selector<Idx0, T, Vals...>::value,
                   selector<Idx1, T, Vals...>::value,
                   selector<Idx2, T, Vals...>::value>;
};
#endif

template <unsigned N, class M>
struct permuter;

template <unsigned Idx0,
          template <unsigned...> class M>
struct permuter<1, M<Idx0>> {
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

template <unsigned Idx0, unsigned Idx1,
          template <unsigned...> class M>
struct permuter<2, M<Idx0, Idx1>> {
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

template <unsigned Idx0, unsigned Idx1, unsigned Idx2,
          template <unsigned...> class M>
struct permuter<3, M<Idx0, Idx1, Idx2>> {
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
        return reorder_gather(cont, permuter::as_array());
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

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
