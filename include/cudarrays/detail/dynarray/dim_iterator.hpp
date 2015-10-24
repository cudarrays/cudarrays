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
#ifndef CUDARRAYS_DETAIL_DYNARRAY_DIM_ITERATOR_HPP_
#define CUDARRAYS_DETAIL_DYNARRAY_DIM_ITERATOR_HPP_

#include "../../common.hpp"
#include "../utils/stl.hpp"

#include "iterator.hpp"

namespace cudarrays {

template <typename Array, bool Const, unsigned Dim, typename Traits>
class array_dim_iterator_detail {
    using traits_base = Traits;
public:
    static constexpr auto is_const    = Const;
    static constexpr auto is_last_dim = (Dim == Array::dimensions - 1);
    static constexpr auto current_dim = Dim;

    using difference_type = typename traits_base::difference_type;
    using      value_type = typename traits_base::value_type;
    using       reference = typename traits_base::reference;
    using         pointer = typename traits_base::pointer;

    using array_reference = typename std::conditional<Const, const Array &, Array &>::type;
    using   array_pointer = typename std::conditional<Const, const Array *, Array *>::type;

protected:
    // Empty iterator
    inline
    array_dim_iterator_detail() :
        parent_(NULL)
    {
    }

    // Initialize iterator
    inline
    array_dim_iterator_detail(array_reference parent, array_index_t idx) :
        parent_(&parent)
    {
        idx_[Dim] = idx;
    }

    // Initialize iterator with previous offsets
    inline
    array_dim_iterator_detail(array_reference parent, const array_index_t off[Dim], array_index_t curr) :
        parent_(&parent)
    {
        std::copy(off, off + Dim, idx_);
        idx_[Dim] = curr;
    }

    inline
    void inc(array_index_t off)
    {
        idx_[Dim] += off;
    }

    inline
    void dec(array_index_t off)
    {
        idx_[Dim] -= off;
    }

    inline
    bool less_than(const array_dim_iterator_detail &it) const
    {
        return less<false>(it);
    }

    inline
    bool less_eq_than(const array_dim_iterator_detail &it) const
    {
        return less<true>(it);
    }

    inline
    bool greater_than(const array_dim_iterator_detail &it) const
    {
        return greater<false>(it);
    }

    inline
    bool greater_eq_than(const array_dim_iterator_detail &it) const
    {
        return greater<true>(it);
    }

    inline
    difference_type subtract(const array_dim_iterator_detail &it) const
    {
        return idx_[Dim] - it.idx_[Dim];
    }

    array_pointer parent_;
    array_index_t idx_[Dim + 1];

private:
    template <bool Equal>
    inline
    bool less(const array_dim_iterator_detail &it) const
    {
        if (Equal)
            return idx_[Dim] <= it.idx_[Dim];
        else
            return idx_[Dim] <  it.idx_[Dim];
    }

    template <bool Equal>
    inline
    bool greater(const array_dim_iterator_detail &it) const
    {
        if (Equal)
            return idx_[Dim] >= it.idx_[Dim];
        else
            return idx_[Dim] >  it.idx_[Dim];
    }
};

template <bool Const, typename PrevIterator, typename Iterator, typename ConstIterator>
class array_dim_iterator_facade_helper {
    static constexpr auto  current_dim = Iterator::current_dim;
    static constexpr auto is_first_dim = current_dim == 0;

    using       iterator = Iterator;
    using const_iterator = ConstIterator;

    using       reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    union {
        typename Iterator::array_pointer parent_;
        PrevIterator *prev_;
    };

    typename Iterator::array_reference
    inline
    get_parent()
    {
        if (is_first_dim)
            return *parent_;
        else
            return *prev_->get_parent();
    }

    typename Iterator::array_reference
    inline
    get_parent() const
    {
        if (is_first_dim)
            return *parent_;
        else
            return *prev_->get_parent();
    }

public:
    template <typename U = array_dim_iterator_facade_helper>
    inline
    array_dim_iterator_facade_helper(utils::enable_if_t<!U::is_first_dim, PrevIterator> &prev) :
        prev_{&prev}
    {
    }

    template <typename U = array_dim_iterator_facade_helper>
    inline
    array_dim_iterator_facade_helper(utils::enable_if_t<U::is_first_dim, typename Iterator::array_reference> parent) :
        parent_{&parent}
    {
    }

    inline
    iterator begin()
    {
        if (is_first_dim)
            return iterator{this->get_parent(), 0};
        else
            return iterator{this->get_parent(), this->prev_->get_idx(), 0};
    }

    inline
    const_iterator begin() const
    {
        return cbegin();
    }

    inline
    const_iterator cbegin() const
    {
        if (is_first_dim)
            return const_iterator{this->get_parent(), 0};
        else
            return const_iterator{this->get_parent(), this->prev_->get_idx(), 0};
    }

    inline
    iterator end()
    {
        array_index_t current = this->get_parent().template dim<current_dim>();
        if (is_first_dim)
            return iterator{this->get_parent(), current};
        else
            return iterator{this->get_parent(), this->prev_->get_idx(), current};
    }

    inline
    const_iterator end() const
    {
        return cend();
    }

    inline
    const_iterator cend() const
    {
        array_index_t current = this->get_parent().template dim<current_dim>();
        if (is_first_dim)
            return const_iterator{this->get_parent(), current};
        else
            return const_iterator{this->get_parent(), this->prev_->get_idx(), current};
    }

    inline
    reverse_iterator rbegin()
    {
        if (is_first_dim)
            return reverse_iterator{iterator{this->get_parent(), this->get_parent().template dim<current_dim>() - 1}};
        else
            return reverse_iterator{iterator{this->get_parent(), this->prev_->get_idx(), this->get_parent().template dim<current_dim>() - 1}};
    }

    inline
    const_reverse_iterator rbegin() const
    {
        return crbegin();
    }

    inline
    const_reverse_iterator crbegin() const
    {
        if (is_first_dim)
            return const_reverse_iterator{const_iterator{this->get_parent(), this->get_parent().template dim<current_dim>() - 1}};
        else
            return const_reverse_iterator{const_iterator{this->get_parent(), this->prev_->get_idx(), this->get_parent().template dim<current_dim>() - 1}};
    }

    inline
    reverse_iterator rend()
    {
        if (is_first_dim)
            return reverse_iterator{iterator{this->get_parent(), -1}};
        else
            return reverse_iterator{iterator{this->get_parent(), this->prev_->get_idx(), -1}};
    }

    inline
    const_reverse_iterator rend() const
    {
        return crend();
    }

    inline
    const_reverse_iterator crend() const
    {
        if (is_first_dim)
            return const_reverse_iterator{const_iterator{this->get_parent(), -1}};
        else
            return const_reverse_iterator{const_iterator{this->get_parent(), this->prev_->get_idx(), -1}};
    }
};

template <typename Array, bool Const, unsigned Dim, bool Last>
class array_dim_iterator;

template <typename Array, bool Const, unsigned Dim>
class array_dim_iterator<Array, Const, Dim, true> :
    public array_dim_iterator_detail<Array, Const, Dim,
                                     array_iterator_traits<Array, Const>> {
    using array_type = Array;

    using iterator_traits_base = array_iterator_traits<array_type, Const>;

    using      parent_type = array_dim_iterator_detail<array_type, Const, Dim, iterator_traits_base>;

public:

    using   array_reference = typename parent_type::array_reference;
    using   array_pointer   = typename parent_type::array_pointer;

    using   difference_type = typename iterator_traits_base::difference_type;
    using        value_type = typename iterator_traits_base::value_type;
    using         reference = typename iterator_traits_base::reference;
    using           pointer = typename iterator_traits_base::pointer;
    using iterator_category = std::random_access_iterator_tag;

    inline
    array_dim_iterator(array_reference parent, array_index_t idx) :
        parent_type(parent, idx)
    {
    }

    inline
    array_dim_iterator(array_reference parent, const array_index_t off[Dim], array_index_t idx) :
        parent_type(parent, off, idx)
    {
    }

    inline reference
    operator*() const
    {
        using dereference_type = array_iterator_dereference<array_type, Const>;
        return dereference_type::unwrap(*this->parent_, this->idx_);
    }

    inline
    pointer operator->() const
    {
        return &(operator*());
    }

    inline bool operator==(array_dim_iterator it) const
    {
        return parent_type::parent_ == it.parent_ && parent_type::idx_[Dim] == it.idx_[Dim];
    }

    inline bool operator!=(array_dim_iterator it) const
    {
        return !(*this == it);
    }

    inline array_dim_iterator &operator++()
    {
        parent_type::inc(1);
        return *this;
    }

    inline array_dim_iterator operator++(int)
    {
        array_dim_iterator res{*this};
        ++(*this);
        return res;
    }

    inline array_dim_iterator &operator--()
    {
        parent_type::dec(1);
        return *this;
    }

    inline array_dim_iterator operator--(int)
    {
        array_dim_iterator res{*this};
        --(*this);
        return res;
    }

    inline array_dim_iterator &operator+=(difference_type off)
    {
        parent_type::inc(off);
        return *this;
    }

    inline array_dim_iterator &operator-=(difference_type off)
    {
        parent_type::dec(off);
        return *this;
    }

    inline array_dim_iterator operator+(difference_type inc) const
    {
        array_dim_iterator res{*this};
        res += inc;
        return res;
    }

    inline array_dim_iterator operator-(difference_type dec) const
    {
        array_dim_iterator res{*this};
        res -= dec;
        return res;
    }

    inline difference_type operator-(const array_dim_iterator &i) const
    {
        return parent_type::subtract(i);
    }

    inline bool operator<(const array_dim_iterator &i) const
    {
        return parent_type::less_than(i);
    }

    inline bool operator<=(const array_dim_iterator &i) const
    {
        return parent_type::less_eq_than(i);
    }

    inline bool operator>(const array_dim_iterator &i) const
    {
        return parent_type::greater_than(i);
    }

    inline bool operator>=(const array_dim_iterator &i) const
    {
        return parent_type::greater_eq_than(i);
    }

    inline value_type &operator[](difference_type i)
    {
        return *((*this) + i);
    }
};

template <typename Array, bool Const, unsigned Dim>
class array_dim_iterator<Array, Const, Dim, false> :
    public array_dim_iterator_detail<Array, Const, Dim,
                                     iterator_traits<
                                         array_dim_iterator_facade_helper<
                                             Const,
                                             array_dim_iterator<Array, Const, Dim, false>,
                                             array_dim_iterator<Array, false, Dim + 1, Array::dimensions - 1 == Dim + 1>,
                                             array_dim_iterator<Array, true,  Dim + 1, Array::dimensions - 1 == Dim + 1>
                                         >,
                                         Const
                                    >
           > {
    using  array_type = Array;

    using iterator_traits_base = iterator_traits<
                                    array_dim_iterator_facade_helper<
                                        Const,
                                        array_dim_iterator<array_type, Const, Dim, false>,
                                        array_dim_iterator<array_type, false, Dim + 1, array_type::dimensions - 1 == Dim + 1>,
                                        array_dim_iterator<array_type, true,  Dim + 1, array_type::dimensions - 1 == Dim + 1>
                                    >,
                                    Const
                                 >;

    using parent_type = array_dim_iterator_detail<array_type, Const, Dim, iterator_traits_base>;

public:

    using   array_reference = typename parent_type::array_reference;
    using   array_pointer   = typename parent_type::array_pointer;

    using   difference_type = typename iterator_traits_base::difference_type;
    using        value_type = typename iterator_traits_base::value_type;
    using         reference = const value_type &;
    using           pointer = const value_type *;
    using iterator_category = std::random_access_iterator_tag;


    inline
    const array_index_t *get_idx() const
    {
        return this->idx_;
    }

    template <typename U = array_dim_iterator>
    inline
    utils::enable_if_t<!U::is_const, array_pointer>
    get_parent()
    {
        return this->parent_;
    }

    template <typename U = array_dim_iterator>
    inline
    utils::enable_if_t<U::is_const, array_pointer>
    get_parent() const
    {
        return this->parent_;
    }


private:
    value_type next_;
    friend value_type;

public:
    inline
    array_dim_iterator(array_reference parent, array_index_t idx) :
        parent_type(parent, idx),
        next_{*this}
    {
    }

    inline
    array_dim_iterator(array_reference parent, const array_index_t off[Dim], array_index_t idx) :
        parent_type(parent, off, idx),
        next_{*this}
    {
    }

    inline reference operator*() const
    {
        return next_;
    }

    inline pointer operator->() const
    {
        return &(operator*());
    }

    inline bool operator==(array_dim_iterator it) const
    {
        return parent_type::parent_ == it.parent_ && parent_type::idx_[Dim] == it.idx_[Dim];
    }

    inline bool operator!=(array_dim_iterator it) const
    {
        return !(*this == it);
    }

    inline array_dim_iterator &operator++()
    {
        parent_type::inc(1);
        return *this;
    }

    inline array_dim_iterator operator++(int)
    {
        array_dim_iterator res{*this};
        ++(*this);
        return res;
    }

    inline array_dim_iterator &operator--()
    {
        parent_type::dec(1);
        return *this;
    }

    inline array_dim_iterator operator--(int)
    {
        array_dim_iterator res{*this};
        --(*this);
        return res;
    }

    inline array_dim_iterator &operator+=(difference_type off)
    {
        parent_type::inc(off);
        return *this;
    }

    inline array_dim_iterator &operator-=(difference_type off)
    {
        parent_type::dec(off);
        return *this;
    }

    inline array_dim_iterator operator+(difference_type inc) const
    {
        array_dim_iterator res{*this};
        res += inc;
        return res;
    }

    inline array_dim_iterator operator-(difference_type dec) const
    {
        array_dim_iterator res{*this};
        res -= dec;
        return res;
    }

    inline difference_type operator-(const array_dim_iterator &i) const
    {
        return parent_type::subtract(i);
    }

    inline bool operator<(const array_dim_iterator &i) const
    {
        return parent_type::less_than(i);
    }

    inline bool operator<=(const array_dim_iterator &i) const
    {
        return parent_type::less_eq_than(i);
    }

    inline bool operator>(const array_dim_iterator &i) const
    {
        return parent_type::greater_than(i);
    }

    inline bool operator>=(const array_dim_iterator &i) const
    {
        return parent_type::greater_eq_than(i);
    }

    inline value_type &operator[](difference_type i)
    {
        return *((*this) + i);
    }
};


template <typename Array, bool Const>
using array_dim_iterator_facade =
    array_dim_iterator_facade_helper<
        Const,
        array_dim_iterator<Array, false, 0, false>,
        array_dim_iterator<Array, false, 0, 1 == Array::dimensions>,
        array_dim_iterator<Array, true,  0, 1 == Array::dimensions>
    >;
}

#endif // CUDARRAYS_DETAIL_DYNARRAY_DIM_ITERATOR_HPP_

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
