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
#ifndef CUDARRAYS_DETAIL_DYNARRAY_BASE_HPP_
#define CUDARRAYS_DETAIL_DYNARRAY_BASE_HPP_

#include "../../common.hpp"
#include "../../host.hpp"
#include "../../storage.hpp"

#include "indexing.hpp"

namespace cudarrays {

template <unsigned Dims>
using extents = std::array<array_size_t, Dims>;

template <typename... T>
auto make_extents(T... values) -> extents<sizeof...(T)>
{
    return extents<sizeof...(T)>{array_size_t(values)...};
}

struct align_t {
    array_size_t alignment;
    array_size_t position;

    explicit align_t(array_size_t _alignment = 0,
                     array_index_t _position = 0) :
        alignment(_alignment),
        position(_position)
    {
    }

    std::tuple<array_size_t, array_size_t>
    align(array_size_t dim) const
    {
        array_size_t offset;
        array_size_t sizeAlign;

        offset = 0;
        if (alignment > 1) {
            if (position > alignment) {
                offset = utils::round_next(position, alignment) - position;
            } else if (position > 0) {
                offset = alignment - position;
            }
            sizeAlign = utils::round_next(dim + offset, alignment);
        } else {
            sizeAlign = dim;
        }

        return std::make_tuple(offset, sizeAlign);
    }
};

template <typename T, unsigned Dims>
class dim_manager {
public:
    static constexpr unsigned FirstDim = 3 - Dims;

    static constexpr unsigned DimIdxZ = 0 - FirstDim;
    static constexpr unsigned DimIdxY = 1 - FirstDim;
    static constexpr unsigned DimIdxX = 2 - FirstDim;

private:
    array_size_t sizes_[Dims];
    array_size_t *sizesAlign_;
    array_size_t offsAlign_[Dims - 1];

    array_size_t elems_;
    array_size_t offset_;

    array_size_t elemsAlign_;

public:
    __host__
    dim_manager(const extents<Dims> &extents,
                const align_t &align)
    {
        ASSERT(extents.size() == Dims);

        sizesAlign_ = new array_size_t[Dims];

        // Initialize array sizes
        utils::copy(extents, sizes_);
        std::copy(extents.begin(), extents.end(), sizesAlign_);

        // Compute offset and aligned size of the lowest order dimension
        std::tie(offset_, sizesAlign_[Dims - 1]) = align.align(extents[Dims - 1]);

        // Fill offsets' array
        array_size_t nextOffAlign = 1;
        for (int i = Dims - 1; i > 0; --i) {
            nextOffAlign     *= sizesAlign_[i];
            offsAlign_[i - 1] = nextOffAlign;
        }

        // Compute number of elements
        elems_      = utils::accumulate(sizes_, 1, std::multiplies<array_size_t>());
        elemsAlign_ = std::accumulate(sizesAlign_, sizesAlign_ + Dims,
                                      1, std::multiplies<array_size_t>());
    }

    __host__
    ~dim_manager()
    {
        delete [] sizesAlign_;
    }

    __host__ __device__
    inline
    array_size_t get_elems() const
    {
        return elems_;
    }

    __host__ __device__
    inline
    array_size_t get_elems_align() const
    {
        return elemsAlign_;
    }

    using offs_align_type = array_size_t[Dims - 1];
    __host__ __device__
    inline
    const offs_align_type &get_offs_align() const
    {
        return offsAlign_;
    }

    __host__ __device__
    inline
    array_size_t offset() const
    {
        return offset_;
    }

    __host__ __device__
    inline
    array_size_t dim(unsigned dim) const
    {
        return this->sizes_[dim];
    }

    __host__ __device__
    inline
    array_size_t dim_align(unsigned dim) const
    {
        return this->sizesAlign_[dim];
    }

    using sizes_type = array_size_t[Dims];
    __host__ __device__
    inline
    const sizes_type &dims() const
    {
        return sizes_;
    }

    __host__ __device__
    inline
    const sizes_type &dims_align() const
    {
        return (const sizes_type &) sizesAlign_;
    }

    CUDARRAYS_TESTED(storage_test, dim_manager)
    CUDARRAYS_TESTED(storage_test, dim_manager_get_dim)
};

template <typename T, unsigned Dims>
class dynarray_base {
public:
    using  dim_manager_type = dim_manager<T, Dims>;
    dynarray_base(const extents<Dims> &extents,
                  const align_t &align) :
        dimManager_(extents, align)
    {
    }

    virtual ~dynarray_base()
    {
    }

    inline __host__ __device__
    const dim_manager_type &
    get_dim_manager() const
    {
        return dimManager_;
    }

    virtual void set_current_gpu(unsigned /*idx*/) {}
    virtual void to_device(host_storage<T> &host) = 0;
    virtual void to_host(host_storage<T> &host) = 0;

private:
    dim_manager_type dimManager_;
};

template <typename T, unsigned Dims, storage_tag StorageImpl, typename PartConf>
class dynarray_storage;

}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
