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
#ifndef CUDARRAYS_UTIL_HPP_
#define CUDARRAYS_UTIL_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <numeric>
#include <sstream>

#include "compiler.hpp"
#include "detail/utils/integral_iterator.hpp"
#include "detail/utils/log.hpp"

namespace std {

static int *
begin(int(&)[0])
{
    return nullptr;
}

static int *
end(int(&)[0])
{
    return nullptr;
}

static unsigned *
begin(unsigned(&)[0])
{
    return nullptr;
}

static unsigned *
end(unsigned(&)[0])
{
    return nullptr;
}

static long int *
begin(long int(&)[0])
{
    return nullptr;
}

static long int *
end(long int(&)[0])
{
    return nullptr;
}

static long unsigned *
begin(long unsigned(&)[0])
{
    return nullptr;
}

static long unsigned *
end(long unsigned(&)[0])
{
    return nullptr;
}

}

namespace utils {

template <typename C, typename T>
static inline
typename C::difference_type
count(const C &cont, const T &val)
{
    return std::count(std::begin(cont), std::end(cont), val);
}

template <typename C, typename P>
static inline
typename C::difference_type
count_if(const C &cont, P pred)
{
    return std::count_if(std::begin(cont), std::end(cont), pred);
}

template <typename C1, typename C2>
static inline
void
copy(C1 &&in, C2 &&out)
{
    std::copy(std::begin(in), std::end(in), std::begin(out));
}

template <typename C>
static inline
bool
equal(const C &a, const C &b)
{
    return std::equal(std::begin(a), std::end(a), std::begin(b));
}

template <typename C, typename T>
static inline
void
fill(C &cont, T val)
{
    return std::fill(std::begin(cont), std::end(cont), val);
}

template <typename C, typename T, typename BinaryOperation>
static inline
T
accumulate(const C &cont, T init, BinaryOperation op)
{
    return std::accumulate(std::begin(cont), std::end(cont), init, op);
}

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
template <typename T, size_t Elems>
struct array_dev {
    T data_[Elems];

    __hostdevice__
    T &operator[](size_t idx)
    {
        return data_[idx];
    }

    __hostdevice__
    constexpr
    const T &operator[](size_t idx) const
    {
        return data_[idx];
    }

    constexpr size_t size() const
    {
        return Elems;
    }
};

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
    T &operator[](size_t idx)
    {
        return data_[idx];
    }

    __hostdevice__
    constexpr
    const T &operator[](size_t idx) const
    {
        return data_[idx];
    }

    __hostdevice__
    constexpr size_t size() const
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
    unsigned dim_index(unsigned orig)
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

template <typename T>
std::string
to_string(T *array, unsigned dims)
{
    std::stringstream str_tmp;

    for (unsigned i = 0; i < dims; ++i) {
        str_tmp << array[i];
        if (i < dims - 1) str_tmp << ", ";
    }

    return str_tmp.str();
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

template <typename T, size_t dims>
std::string
to_string(const T (&array)[dims])
{
    return to_string(array, dims);
}

template <typename T>
std::string
to_string(const T (&array)[0])
{
    return to_string(array, 0);
}

template <typename T, size_t Dims>
std::string
to_string(const std::array<T, Dims> &array)
{
    return to_string(array.data(), Dims);
}

template <typename O>
O &
operator<<(O &out, float2 val)
{
    out << "("
        << val.x << ", " << val.y
        << ")";

    return out;
}

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


#endif

}

#endif // CUDARRAYS_UTIL_HPP_
