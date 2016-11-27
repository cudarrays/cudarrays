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

#include <array>

#include "common.hpp"

#include "cudarrays/detail/dynarray/iterator.hpp"

#include "gtest/gtest.h"

class iterator_test :
    public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}
};

template <typename T, unsigned Dims>
class array_test {
public:
    static constexpr bool has_alignment = 0;

    static constexpr auto dimensions = Dims;

    using value_type      = T;

    using difference_type = cudarrays::array_index_t;

    using iterator       = cudarrays::array_iterator<array_test, false>;
    using const_iterator = cudarrays::array_iterator<array_test, true>;

    using reverse_iterator       = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    array_test(const std::array<cudarrays::array_size_t, Dims> dims) :
        dims_(dims)
    {
        data_ = new T[get_nelems()];
    }

    array_test(const array_test &a) :
        dims_(a.dims_)
    {
        data_ = new T[get_nelems()];

        std::copy(a.data_, a.data_ + get_nelems(), data_);
    }

    array_test(array_test &&a) :
        dims_(a.dims_)
    {
        data_ = a.data_;
        a.data_ = nullptr;
    }

    ~array_test()
    {
        delete [] data_;
    }

    iterator begin()
    {
        return iterator{*this, 0};
    }

    const_iterator begin() const
    {
        return cbegin();
    }

    const_iterator cbegin() const
    {
        return const_iterator{*this, 0};
    }

    reverse_iterator rbegin()
    {
        return reverse_iterator{iterator{*this, cudarrays::array_index_t(get_nelems())}};
    }

    const_reverse_iterator rbegin() const
    {
        return crbegin();
    }

    const_reverse_iterator crbegin() const
    {
        return const_reverse_iterator{const_iterator{*this, cudarrays::array_index_t(get_nelems())}};
    }

    iterator end()
    {
        return iterator{*this, cudarrays::array_index_t(get_nelems())};
    }

    const_iterator end() const
    {
        return cend();
    }

    const_iterator cend() const
    {
        return const_iterator{*this, cudarrays::array_index_t(get_nelems())};
    }

    reverse_iterator rend()
    {
        return reverse_iterator{iterator{*this, 0}};
    }

    const_reverse_iterator rend() const
    {
        return crend();
    }

    const_reverse_iterator crend() const
    {
        return const_reverse_iterator{const_iterator{*this, 0}};
    }

    cudarrays::array_size_t dim(unsigned dim) const
    {
        return dims_[dim];
    }

    cudarrays::array_size_t get_nelems() const
    {
        cudarrays::array_size_t ret;

        ret = std::accumulate(dims_.begin(), dims_.end(), 1,
                              std::multiplies<cudarrays::array_size_t>{});

        return ret;
    }

    value_type &operator()(cudarrays::array_index_t i)
    {
        return data_[i];
    }

    value_type &operator()(cudarrays::array_index_t i, cudarrays::array_index_t j)
    {
        return data_[i * dims_[1] + j];
    }

    value_type &operator()(cudarrays::array_index_t i, cudarrays::array_index_t j, cudarrays::array_index_t k)
    {
        return data_[i * dims_[1] * dims_[2] + j * dims_[2] + k];
    }

    const value_type &operator()(cudarrays::array_index_t i) const
    {
        return data_[i];
    }

    const value_type &operator()(cudarrays::array_index_t i, cudarrays::array_index_t j) const
    {
        return data_[i * dims_[1] + j];
    }

    const value_type &operator()(cudarrays::array_index_t i, cudarrays::array_index_t j, cudarrays::array_index_t k) const
    {
        return data_[i * dims_[1] * dims_[2] + j * dims_[2] + k];
    }

    T *host_addr()
    {
        return data_;
    }

    const T *host_addr() const
    {
        return data_;
    }

private:
    std::array<cudarrays::array_size_t, Dims> dims_;

    T *data_;
};

template <typename T>
using array1d = array_test<T, 1>;
template <typename T>
using array2d = array_test<T, 2>;
template <typename T>
using array3d = array_test<T, 3>;

TEST_F(iterator_test, iterator1d)
{
    array1d<float> a{{100}};;

    float val = 0;
    std::generate(a.host_addr(), a.host_addr() + a.get_nelems(),
                  [&val]() -> float
                  {
                      return val++;
                  });

    val = 0;
    for (auto v : a) {
        ASSERT_EQ(v, val++);
    }

    val = a.get_nelems();
    for (auto it = a.rbegin(); it != a.rend(); ++it) {
        ASSERT_EQ(*it, --val);
    }

    auto it  = a.begin();
    auto end = a.end();

    ASSERT_EQ(it <  end, true);
    ASSERT_EQ(it <= end, true);
    ASSERT_EQ(it >  end, false);
    ASSERT_EQ(it >= end, false);
    ASSERT_EQ(end >  it, true);
    ASSERT_EQ(end >= it, true);
    ASSERT_EQ(end <  it, false);
    ASSERT_EQ(end <= it, false);
    ASSERT_EQ(it + a.get_nelems() == end, true);
    ASSERT_EQ(it + a.get_nelems() != end, false);
    ASSERT_EQ(it + a.get_nelems() <= end, true);
    ASSERT_EQ(it + a.get_nelems() <  end, false);
    ASSERT_EQ(it + a.get_nelems() >= end, true);
    ASSERT_EQ(it + a.get_nelems() >  end, false);
}

TEST_F(iterator_test, iterator2d)
{
    array2d<float> a{{100, 50}};

    float val = 0;
    std::generate(a.host_addr(), a.host_addr() + a.get_nelems(),
                  [&val]() -> float
                  {
                      return val++;
                  });

    val = 0;
    for (auto v : a) {
        ASSERT_EQ(v, val++);
    }

    val = a.get_nelems();
    for (auto it = a.rbegin(); it != a.rend(); ++it) {
        ASSERT_EQ(*it, --val);
    }
    val = a.get_nelems();
    for (auto it = a.crbegin(); it != a.crend(); ++it) {
        ASSERT_EQ(*it, --val);
    }

    auto it  = a.begin();
    auto end = a.end();

    ASSERT_EQ(it <  end, true);
    ASSERT_EQ(it <= end, true);
    ASSERT_EQ(it >  end, false);
    ASSERT_EQ(it >= end, false);
    ASSERT_EQ(end >  it, true);
    ASSERT_EQ(end >= it, true);
    ASSERT_EQ(end <  it, false);
    ASSERT_EQ(end <= it, false);
    ASSERT_EQ(it + a.get_nelems() == end, true);
    ASSERT_EQ(it + a.get_nelems() != end, false);
    ASSERT_EQ(it + a.get_nelems() <= end, true);
    ASSERT_EQ(it + a.get_nelems() <  end, false);
    ASSERT_EQ(it + a.get_nelems() >= end, true);
    ASSERT_EQ(it + a.get_nelems() >  end, false);
}
