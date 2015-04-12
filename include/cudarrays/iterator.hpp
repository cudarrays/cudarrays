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
#ifndef CUDARRAYS_ITERATOR_HPP_
#define CUDARRAYS_ITERATOR_HPP_

#include <iterator>

#include "compiler.hpp"
#include "common.hpp"
#include "utils.hpp"

namespace cudarrays {

template <typename Array, bool Const>
struct iterator_adaptor {
    using array_type = Array;
    static constexpr bool is_const = Const;
};

template <typename Array, bool Const>
struct myiterator_traits;

template <typename Array>
struct myiterator_traits<Array, false> {
    using difference_type = typename Array::difference_type;
    using      value_type = typename Array::value_type;
    using       reference = value_type &;
    using         pointer = value_type *;
};

template <typename Array>
struct myiterator_traits<Array, true> {
    using difference_type = typename Array::difference_type;
    using      value_type = typename Array::value_type;
    using       reference = const value_type &;
    using         pointer = const value_type *;
};

}

namespace std {

template <typename Array, bool Const>
struct iterator_traits<cudarrays::iterator_adaptor<Array, Const> > {
    using traits_base = cudarrays::myiterator_traits<Array, Const>;

    using difference_type = typename traits_base::difference_type;
    using      value_type = typename traits_base::value_type;
    using       reference = typename traits_base::reference;
    using         pointer = typename traits_base::pointer;
};

}

namespace cudarrays {

template <typename Array, bool Const, unsigned Dims>
struct myiterator_dereference;

template <typename Array>
struct myiterator_dereference<Array, true, 1> {
    static inline
    typename myiterator_traits<Array, true>::reference
    get_element(const Array &array, const array_index_t cursor[Array::dimensions])
    {
        return array(cursor[0]);
    }
};

template <typename Array>
struct myiterator_dereference<Array, false, 1> {
    static inline
    typename myiterator_traits<Array, false>::reference
    get_element(Array &array, const array_index_t cursor[Array::dimensions])
    {
        return array(cursor[0]);
    }
};

template <typename Array>
struct myiterator_dereference<Array, true, 2> {
    static inline
    typename myiterator_traits<Array, true>::reference
    get_element(const Array &array, const array_index_t cursor[Array::dimensions])
    {
        return array(cursor[0], cursor[1]);
    }
};

template <typename Array>
struct myiterator_dereference<Array, false, 2> {
    static inline
    typename myiterator_traits<Array, false>::reference
    get_element(Array &array, const array_index_t cursor[Array::dimensions])
    {
        return array(cursor[0], cursor[1]);
    }
};

template <typename Array>
struct myiterator_dereference<Array, true, 3> {
    static inline
    typename myiterator_traits<Array, true>::reference
    get_element(const Array &array, const array_index_t cursor[Array::dimensions])
    {
        return array(cursor[0], cursor[1], cursor[2]);
    }
};

template <typename Array>
struct myiterator_dereference<Array, false, 3> {
    static inline
    typename myiterator_traits<Array, false>::reference
    get_element(Array &array, const array_index_t cursor[Array::dimensions])
    {
        return array(cursor[0], cursor[1], cursor[2]);
    }
};

template <typename Array, bool Const, unsigned Dims>
class myiterator_access_detail {
    using traits_base = std::iterator_traits<cudarrays::iterator_adaptor<Array, Const>>;

    using difference_type = typename traits_base::difference_type;
    using      value_type = typename Array::value_type;

    using value_reference = typename std::conditional<Const, const value_type &, value_type &>::type;
    using value_pointer   = typename std::conditional<Const, const value_type *, value_type *>::type;

    using array_reference = typename std::conditional<Const, const Array &, Array &>::type;
    using array_pointer   = typename std::conditional<Const, const Array *, Array *>::type;

    using dereference_type = myiterator_dereference<Array, Const, Dims>;

public:
    inline
    value_reference operator*() const
    {
        return dereference_type::get_element(*parent_, idx_);
    }

    inline
    value_pointer operator->() const
    {
        return &(operator*());
    }

protected:
    inline
    myiterator_access_detail() :
        parent_(NULL)
    {
        fill(idx_, -1);
    }

    inline
    myiterator_access_detail(array_reference parent) :
        parent_(&parent)
    {
        fill(idx_, 0);
    }

    inline
    myiterator_access_detail(array_reference parent, array_index_t off[Dims]) :
        parent_(&parent)
    {
        for (unsigned dim = 0; dim < Dims; ++dim) {
            idx_[dim] = off[dim];
        }
    }

    template <bool Unit>
    void inc(array_index_t off)
    {
        for (int dim = Dims - 1; dim >= 0; --dim) {
            difference_type i = idx_[dim] + off;
            if (dim > 0 && i >= parent_->get_dim(dim)) {
                // Next iteration will update dim - 1
                if (Unit) {
                    idx_[dim] = 0;
                    off = 1;
                } else {
                    idx_[dim] = i % difference_type(parent_->get_dim(dim));
                    off       = i / difference_type(parent_->get_dim(dim));
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
        for (int dim = Dims - 1; dim >= 0; --dim) {
            difference_type i = idx_[dim] - off;
            if (dim > 0 && i < 0) {
                // Next iteration will update dim - 1
                if (Unit) {
                    idx_[dim] = difference_type(parent_->get_dim(dim)) - 1;
                    off       = 1;
                } else {
                    idx_[dim] = difference_type(parent_->get_dim(dim)) - (i % difference_type(parent_->get_dim(dim)));
                    off       = i / difference_type(parent_->get_dim(dim));
                }
            } else {
                // No underflow, we are done
                idx_[dim] = i;
                break;
            }
        }
    }

    inline
    bool less_than(const myiterator_access_detail &it) const
    {
        return less<false>(it);
    }

    inline
    bool less_eq_than(const myiterator_access_detail &it) const
    {
        return less<true>(it);
    }

    inline
    bool greater_than(const myiterator_access_detail &it) const
    {
        return greater<false>(it);
    }

    inline
    bool greater_eq_than(const myiterator_access_detail &it) const
    {
        return greater<true>(it);
    }

    difference_type subtract(const myiterator_access_detail &it) const
    {
        // TODO: optimize
        difference_type ret = 0;
        difference_type inc = 1;
        for (int dim = Dims - 1; dim >= 0; --dim) {
            ret += (idx_[dim] - it.idx_[dim]) * inc;

            inc *= difference_type(parent_->get_dim(dim));
        }
        return ret;
    }

    array_pointer parent_;
    array_index_t idx_[Dims];

private:
    template <bool Equal>
    bool less(const myiterator_access_detail &it) const
    {
        for (int dim = 0; dim < Dims; ++dim) {
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
    inline
    bool greater(const myiterator_access_detail &it) const
    {
        for (int dim = 0; dim < Dims; ++dim) {
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
class myiterator :
    public myiterator_access_detail<Array, Const, Array::dimensions> {
    using parent_type = myiterator_access_detail<Array, Const, Array::dimensions>;
    using array_reference = typename std::conditional<Const, const Array &, Array &>::type;

public:
    using iterator_traits_base = std::iterator_traits<cudarrays::iterator_adaptor<Array, Const>>;

    using difference_type = typename iterator_traits_base::difference_type;
    using      value_type = typename iterator_traits_base::value_type;
    using       reference = typename iterator_traits_base::reference;
    using         pointer = typename iterator_traits_base::pointer;
    using iterator_category = std::random_access_iterator_tag;

    static constexpr bool is_const = Const;

    inline
    myiterator(array_reference parent) :
        parent_type(parent)
    {
    }

    inline
    myiterator(array_reference parent, array_index_t off[Array::dimensions]) :
        parent_type(parent, off)
    {
    }

    inline bool operator==(myiterator it) const
    {
        return parent_type::parent_ == it.parent_ && utils::equal(parent_type::idx_, it.idx_);
    }

    inline bool operator!=(myiterator it) const
    {
        return !(*this == it);
    }

    inline myiterator &operator++()
    {
        parent_type::template inc<true>(1);
        return *this;
    }

    inline myiterator operator++(int)
    {
        myiterator res(*this);
        ++(*this);
        return res;
    }

    inline myiterator &operator--()
    {
        parent_type::template dec<true>(1);
        return *this;
    }

    inline myiterator operator--(int)
    {
        myiterator res(*this);
        --(*this);
        return res;
    }

    inline myiterator &operator+=(difference_type off)
    {
        parent_type::template inc<false>(off);
        return *this;
    }

    inline myiterator &operator-=(difference_type off)
    {
        parent_type::template dec<false>(off);
        return *this;
    }

    inline myiterator operator+(difference_type inc) const
    {
        myiterator res(*this);
        res += inc;
        return res;
    }

    inline myiterator operator-(difference_type dec) const
    {
        myiterator res(*this);
        res -= dec;
        return res;
    }

    inline difference_type operator-(const myiterator &i) const
    {
        return parent_type::subtract(i);
    }

    inline bool operator<(const myiterator &i) const
    {
        return parent_type::less_than(i);
    }

    inline bool operator<=(const myiterator &i) const
    {
        return parent_type::less_eq_than(i);
    }

    inline bool operator>(const myiterator &i) const
    {
        return parent_type::greater_than(i);
    }

    inline bool operator>=(const myiterator &i) const
    {
        return parent_type::greater_eq_than(i);
    }

    inline value_type &operator[](difference_type i)
    {
        return *((*this) + i);
    }
};

}

#endif // CUDARRAYS_ITERATOR_HPP_

/* im:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
