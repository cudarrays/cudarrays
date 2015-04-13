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
#ifndef CUDARRAYS_DETAIL_UTILS_INTEGRAL_ITERATOR_HPP_
#define CUDARRAYS_DETAIL_UTILS_INTEGRAL_ITERATOR_HPP_

#include <type_traits>

namespace utils {

template <typename Range, typename Range::type step>
class range_iterator {
protected:
    typename Range::type current_;

    inline range_iterator(const typename Range::type &val) :
        current_(val)
    {}

public:
    inline typename Range::type operator*()
    {
        return current_;
    }

    inline range_iterator &operator++()
    {
        current_ += step;
        return *this;
    }

    inline range_iterator operator++(int)
    {
        range_iterator ret(*this);
        current_ += step;
        return ret;
    }

    inline bool operator==(const range_iterator &it) const
    {
        return current_ == it.current_;
    }

    inline bool operator!=(const range_iterator &it) const
    {
        return current_ != it.current_;
    }

    inline bool operator<(const range_iterator &it) const
    {
        return current_ < it.current_;
    }

    inline bool operator>(const range_iterator &it) const
    {
        return current_ > it.current_;
    }

    friend typename Range::myself;
};

template <typename T, T Step>
class irange {
    T min_;
    T max_;
public:
    using type     = T;
    using myself   = irange;
    using iterator = range_iterator<myself, Step>;

    inline irange(T max) :
        min_(0),
        max_(max)
    {
    }

    inline irange(T min, T max) :
        min_(min),
        max_(max)
    {
    }

    inline iterator begin()
    {
        return iterator(min_);
    }

    inline iterator end()
    {
        return iterator(max_);
    }
};

template <typename T, T Step, T Low>
class irange_low {
    T max_;
public:
    using     type = T;
    using   myself = irange_low;
    using iterator = range_iterator<myself, Step>;

    inline explicit irange_low(T max) :
        max_(max)
    {
    }

    inline iterator begin()
    {
        return iterator(Low);
    }

    inline iterator end()
    {
        return iterator(max_);
    }
};


template <typename T, T Step = T(1)>
static inline
irange_low<T, Step, 0>
make_range(T max)
{
    return irange_low<T, Step, 0>{max};
}

template <typename T, typename U, int Step = 1>
static inline
irange
<
    typename std::conditional<
        (std::is_signed<T>::value || std::is_signed<U>::value) || Step < 0,
        typename std::make_signed<typename std::conditional<(sizeof(T) > sizeof(U)), T, U>::type>::type,
        typename std::conditional<(sizeof(T) > sizeof(U)), T, U>::type
    >::type,
    Step
>
make_range(T min, U max)
{
    // Deduce type size
    using _R = typename std::conditional<(sizeof(T) > sizeof(U)), T, U>::type;

    // Deduce signed/unsigned
    using R = typename std::conditional<(std::is_signed<T>::value || std::is_signed<U>::value || Step < 0),
                                        typename std::make_signed<_R>::type,
                                        _R>::type;

    return irange<R, Step>{static_cast<R>(min), static_cast<R>(max)};
}

template <typename T, typename U, unsigned Step = 1>
static inline
irange
<
    typename std::make_signed <
        typename std::conditional<(sizeof(T) > sizeof(U)), T, U>::type
    >::type,
    -int(Step)
>
make_range_dec(T min, U max)
{
    // Deduce type size
    using R = typename std::make_signed<
                                        typename std::conditional<(sizeof(T) > sizeof(U)), T, U>::type
                                       >::type;

    return irange<R, -int(Step)>{static_cast<R>(min), static_cast<R>(max)};
}
}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
