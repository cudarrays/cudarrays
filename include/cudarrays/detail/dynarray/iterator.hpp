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
#ifndef CUDARRAYS_DETAIL_DYNARRAY_ITERATOR_HPP_
#define CUDARRAYS_DETAIL_DYNARRAY_ITERATOR_HPP_

#include <iterator>

#include "../../common.hpp"
#include "../utils/integral_iterator.hpp"
#include "../utils/stl.hpp"

namespace cudarrays {

template <typename Array, bool Const>
struct array_iterator_traits {
    using difference_type = typename Array::difference_type;
    using      value_type = typename Array::value_type;
    using       reference = typename std::conditional<Const, const value_type &, value_type &>::type;
    using         pointer = typename std::conditional<Const, const value_type *, value_type *>::type;
};

template <typename ValueType, bool Const, typename DiffType = std::ptrdiff_t>
struct iterator_traits {
    using      value_type = ValueType;
    using difference_type = DiffType;
    using       reference = typename std::conditional<Const, const value_type &, value_type &>::type;
    using         pointer = typename std::conditional<Const, const value_type *, value_type *>::type;
};

template <typename Array, bool Const, unsigned Current = Array::dimensions - 1>
struct array_iterator_dereference {
    using array_reference = typename std::conditional<Const, const Array &, Array &>::type;
    using       next_type = array_iterator_dereference<Array, Const, Current - 1>;

    template <typename... IdxType>
    static inline
    typename array_iterator_traits<Array, Const>::reference
    unwrap(array_reference array, const array_index_t cursor[Array::dimensions], IdxType... idxs)
    {
        return next_type::unwrap(array, cursor, cursor[Current], idxs...);
    }
};

template <typename Array, bool Const>
struct array_iterator_dereference<Array, Const, 0> {
    using array_reference = typename std::conditional<Const, const Array &, Array &>::type;

    template <typename... IdxType>
    static inline
    typename array_iterator_traits<Array, Const>::reference
    unwrap(array_reference array, const array_index_t cursor[Array::dimensions], IdxType... idxs)
    {
        return array(cursor[0], idxs...);
    }
};

template <typename Array, bool Const, bool IsAligned = Array::has_alignment>
class array_iterator_access_detail;

template <typename Array, bool Const>
class array_iterator_access_detail<Array, Const, true> {
    using     traits_base = array_iterator_traits<Array, Const>;

public:
    using difference_type = typename traits_base::difference_type;
    using      value_type = typename traits_base::value_type;
    using       reference = typename traits_base::reference;
    using         pointer = typename traits_base::pointer;

    using array_reference = typename std::conditional<Const, const Array &, Array &>::type;
    using   array_pointer = typename std::conditional<Const, const Array *, Array *>::type;

    using dereference_type = array_iterator_dereference<Array, Const>;

    inline
    reference operator*() const
    {
        return dereference_type::unwrap(*parent_, idx_);
    }

    inline
    pointer operator->() const
    {
        return &(operator*());
    }

protected:
    inline
    array_iterator_access_detail() :
        parent_(NULL)
    {
        fill(idx_, -1);
    }

    inline
    array_iterator_access_detail(array_reference parent, array_index_t off[Array::dimensions]) :
        parent_(&parent)
    {
        std::copy(off, off + Array::dimensions, idx_);
    }

    template <bool Unit>
    void inc(array_index_t off)
    {
        for (int dim = Array::dimensions - 1; dim >= 0; --dim) {
            difference_type i = idx_[dim] + off;
            if (dim > 0 && i >= difference_type(parent_->dim(dim))) {
                // Next iteration will update dim - 1
                if (Unit) {
                    idx_[dim] = 0;
                    off = 1;
                } else {
                    idx_[dim] = i % difference_type(parent_->dim(dim));
                    off       = i / difference_type(parent_->dim(dim));
                }
            } else {
                // No overflow, we are done
                idx_[dim] = i;
                break;
            }
        }
    }

    template <bool Unit>
    void dec(array_index_t off)
    {
        for (int dim = Array::dimensions - 1; dim >= 0; --dim) {
            difference_type i = idx_[dim] - off;
            if (dim > 0 && i < 0) {
                // Next iteration will update dim - 1
                if (Unit) {
                    idx_[dim] = difference_type(parent_->dim(dim)) - 1;
                    off       = 1;
                } else {
                    idx_[dim] = difference_type(parent_->dim(dim)) - (i % difference_type(parent_->dim(dim)));
                    off       = i / difference_type(parent_->dim(dim));
                }
            } else {
                // No underflow, we are done
                idx_[dim] = i;
                break;
            }
        }
    }

    inline
    bool equal(const array_iterator_access_detail &it) const
    {
        return utils::equal(idx_, it.idx_);
    }

    inline
    bool less_than(const array_iterator_access_detail &it) const
    {
        return less<false>(it);
    }

    inline
    bool less_eq_than(const array_iterator_access_detail &it) const
    {
        return less<true>(it);
    }

    inline
    bool greater_than(const array_iterator_access_detail &it) const
    {
        return greater<false>(it);
    }

    inline
    bool greater_eq_than(const array_iterator_access_detail &it) const
    {
        return greater<true>(it);
    }

    difference_type subtract(const array_iterator_access_detail &it) const
    {
        // TODO: optimize
        difference_type ret = 0;
        difference_type inc = 1;
        for (int dim = Array::dimensions - 1; dim >= 0; --dim) {
            ret += (idx_[dim] - it.idx_[dim]) * inc;

            inc *= difference_type(parent_->dim(dim));
        }
        return ret;
    }

    array_pointer parent_;
    array_index_t idx_[Array::dimensions];

private:
    template <bool Equal>
    bool less(const array_iterator_access_detail &it) const
    {
        for (auto dim : utils::make_range(Array::dimensions)) {
            if (idx_[dim] > it.idx_[dim]) {
                return false;
            } else if (idx_[dim] < it.idx_[dim]) {
                return true;
            }
            // If equal, keep comparing
        }
        return Equal;
    }

    template <bool Equal>
    bool greater(const array_iterator_access_detail &it) const
    {
        for (auto dim : utils::make_range(Array::dimensions)) {
            if (idx_[dim] < it.idx_[dim]) {
                return false;
            } else if (idx_[dim] > it.idx_[dim]) {
                return true;
            }
            // If equal, keep comparing
        }
        return Equal;
    }

    CUDARRAYS_TESTED(iterator_test, iterator1d)
    CUDARRAYS_TESTED(iterator_test, iterator2d)
};

template <typename Array, bool Const>
class array_iterator_access_detail<Array, Const, false> {
    using     traits_base = array_iterator_traits<Array, Const>;

public:
    using difference_type = typename traits_base::difference_type;
    using      value_type = typename traits_base::value_type;
    using       reference = typename traits_base::reference;
    using         pointer = typename traits_base::pointer;

    using array_reference = typename std::conditional<Const, const Array &, Array &>::type;
    using   array_pointer = typename std::conditional<Const, const Array *, Array *>::type;

    inline
    reference operator*() const
    {
        return ((pointer) parent_->host_addr())[idx_];
    }

    inline
    pointer operator->() const
    {
        return &(operator*());
    }

protected:
    inline
    array_iterator_access_detail() :
        parent_(NULL),
        idx_{-1}
    {
    }

    inline
    array_iterator_access_detail(array_reference parent, array_index_t off) :
        parent_(&parent),
        idx_(off)
    {
    }

    template <bool Unit>
    inline
    void inc(array_index_t off)
    {
        idx_ += off;
    }

    template <bool Unit>
    inline
    void dec(array_index_t off)
    {
        idx_ -= off;
    }

    inline
    bool equal(const array_iterator_access_detail &it) const
    {
        return idx_ == it.idx_;
    }

    inline
    bool less_than(const array_iterator_access_detail &it) const
    {
        return less<false>(it);
    }

    inline
    bool less_eq_than(const array_iterator_access_detail &it) const
    {
        return less<true>(it);
    }

    inline
    bool greater_than(const array_iterator_access_detail &it) const
    {
        return greater<false>(it);
    }

    inline
    bool greater_eq_than(const array_iterator_access_detail &it) const
    {
        return greater<true>(it);
    }

    inline
    difference_type subtract(const array_iterator_access_detail &it) const
    {
        return idx_ - it.idx_;
    }

    array_pointer parent_;
    array_index_t idx_;

private:
    template <bool Equal>
    inline
    bool less(const array_iterator_access_detail &it) const
    {
        if (Equal)
            return idx_ <= it.idx_;
        else
            return idx_ < it.idx_;
    }

    template <bool Equal>
    inline
    bool greater(const array_iterator_access_detail &it) const
    {
        if (Equal)
            return idx_ >= it.idx_;
        else
            return idx_ > it.idx_;
    }
};


template <typename Array, bool Const>
class array_iterator :
    public array_iterator_access_detail<Array, Const> {
    using  array_type = Array;
    using parent_type = array_iterator_access_detail<Array, Const>;

public:
    using array_reference = typename parent_type::array_reference;

    using iterator_traits_base = array_iterator_traits<Array, Const>;

    using difference_type = typename iterator_traits_base::difference_type;
    using      value_type = typename iterator_traits_base::value_type;
    using       reference = typename iterator_traits_base::reference;
    using         pointer = typename iterator_traits_base::pointer;
    using iterator_category = std::random_access_iterator_tag;

    static constexpr bool is_const = Const;

    inline
    array_iterator() :
        parent_type()
    {
    }

    template <typename U = Array>
    inline
    array_iterator(utils::enable_if_t<U::has_alignment, array_reference> parent, array_index_t off[Array::dimensions]) :
        parent_type(parent, off)
    {
    }

    template <typename U = Array>
    inline
    array_iterator(utils::enable_if_t<!U::has_alignment, array_reference> parent, array_index_t off) :
        parent_type(parent, off)
    {
    }

    inline bool operator==(array_iterator it) const
    {
        return parent_type::parent_ == it.parent_ && parent_type::equal(it);
    }

    inline bool operator!=(array_iterator it) const
    {
        return !(*this == it);
    }

    inline array_iterator &operator++()
    {
        parent_type::template inc<true>(1);
        return *this;
    }

    inline array_iterator operator++(int)
    {
        array_iterator res(*this);
        ++(*this);
        return res;
    }

    inline array_iterator &operator--()
    {
        parent_type::template dec<true>(1);
        return *this;
    }

    inline array_iterator operator--(int)
    {
        array_iterator res(*this);
        --(*this);
        return res;
    }

    inline array_iterator &operator+=(difference_type off)
    {
        parent_type::template inc<false>(off);
        return *this;
    }

    inline array_iterator &operator-=(difference_type off)
    {
        parent_type::template dec<false>(off);
        return *this;
    }

    inline array_iterator operator+(difference_type inc) const
    {
        array_iterator res(*this);
        res += inc;
        return res;
    }

    inline array_iterator operator-(difference_type dec) const
    {
        array_iterator res(*this);
        res -= dec;
        return res;
    }

    inline difference_type operator-(const array_iterator &i) const
    {
        return parent_type::subtract(i);
    }

    inline bool operator<(const array_iterator &i) const
    {
        return parent_type::less_than(i);
    }

    inline bool operator<=(const array_iterator &i) const
    {
        return parent_type::less_eq_than(i);
    }

    inline bool operator>(const array_iterator &i) const
    {
        return parent_type::greater_than(i);
    }

    inline bool operator>=(const array_iterator &i) const
    {
        return parent_type::greater_eq_than(i);
    }

    inline value_type &operator[](difference_type i)
    {
        return *((*this) + i);
    }
};

template <typename T, bool Const, bool IsAligned = T::has_alignment>
class array_iterator_facade
{
public:
    static constexpr bool is_const = Const;
    using   iterator_type = array_iterator<T, Const>;
    using array_reference = typename iterator_type::array_reference;

    using array_type = T;
    //
    // Iterator interface
    //
    using       iterator = array_iterator<array_type, false>;
    using const_iterator = array_iterator<array_type, true>;

    using       reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    inline
    array_iterator_facade(array_reference a) :
        parent_{a}
    {
    }

    template <typename U = T>
    inline
    utils::enable_if_t<U::has_alignment, iterator>
    begin()
    {
        array_index_t dims[array_type::dimensions];
        std::fill(dims, dims + array_type::dimensions, 0);
        return iterator{parent_, dims};
    }

    template <typename U = T>
    inline
    utils::enable_if_t<!U::has_alignment, iterator>
    begin()
    {
        return iterator{parent_, 0};
    }

    inline
    const_iterator begin() const
    {
        return cbegin();
    }

    template <typename U = T>
    inline
    utils::enable_if_t<U::has_alignment, const_iterator>
    cbegin() const
    {
        array_index_t dims[array_type::dimensions];
        std::fill(dims, dims + array_type::dimensions, 0);
        return const_iterator{parent_, dims};
    }

    template <typename U = T>
    inline
    utils::enable_if_t<!U::has_alignment, const_iterator>
    cbegin() const
    {
        return const_iterator{parent_, 0};
    }

    inline
    reverse_iterator rbegin()
    {
        return reverse_iterator(end());
    }

    inline
    const_reverse_iterator rbegin() const
    {
        return crbegin();
    }

    inline
    const_reverse_iterator crbegin() const
    {
        return const_reverse_iterator(cend());
    }

    template <typename U = T>
    inline
    utils::enable_if_t<U::has_alignment, iterator>
    end()
    {
        array_index_t dims[array_type::dimensions];
        dims[0] = parent_.dim(0);
        if (array_type::dimensions > 1) {
            std::fill(dims + 1, dims + array_type::dimensions, 0);
        }
        return iterator(parent_, dims);
    }

    template <typename U = T>
    inline
    utils::enable_if_t<!U::has_alignment, iterator>
    end()
    {
        return iterator(parent_, parent_.get_dim_manager().get_elems_align());
    }

    inline
    const_iterator end() const
    {
        return cend();
    }

    template <typename U = T>
    inline
    utils::enable_if_t<U::has_alignment, const_iterator>
    cend() const
    {
        array_index_t dims[array_type::dimensions];
        dims[0] = parent_.dim(0);
        if (array_type::dimensions > 1) {
            std::fill(dims + 1, dims + array_type::dimensions, 0);
        }
        return const_iterator(parent_, dims);
    }

    template <typename U = T>
    inline
    utils::enable_if_t<!U::has_alignment, const_iterator>
    cend() const
    {
        return const_iterator(parent_, parent_.get_dim_manager().get_elems_align());
    }

    inline
    reverse_iterator rend()
    {
        return reverse_iterator(begin());
    }

    inline
    const_reverse_iterator rend() const
    {
        return crend();
    }

    inline
    const_reverse_iterator crend() const
    {
        return const_reverse_iterator(cbegin());
    }

private:
    array_reference parent_;
};

}

#endif // CUDARRAYS_ITERATOR_HPP_

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
