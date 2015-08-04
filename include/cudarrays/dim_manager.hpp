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
#ifndef CUDARRAYS_DIM_MANAGER_HPP_
#define CUDARRAYS_DIM_MANAGER_HPP_

#include "config.hpp"
#include "utils.hpp"

namespace cudarrays {

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
    array_size_t sizeAlign_; // Lowest-order dimension
    array_size_t strides_[Dims - 1];

    array_size_t offset_;

    array_size_t elemsAlign_;

public:
    __host__
    dim_manager(const extents<Dims> &extents,
                const align_t &align)
    {
        ASSERT(extents.size() == Dims);

        // Initialize array sizes
        utils::copy(extents, sizes_);

        // Compute offset and aligned size of the lowest order dimension
        std::tie(offset_, sizeAlign_) = align.align(extents[Dims - 1]);

        // Fill offsets' array
        array_size_t nextStride = sizeAlign_;
        for (int i = int(Dims) - 2; i > 0; --i) {
            strides_[i]   = nextStride;
            nextStride *= sizes_[i];
        }

        // Compute number of elements
        elemsAlign_ = nextStride;
    }

    __host__ __device__
    inline
    array_size_t get_elems_align() const
    {
        return elemsAlign_;
    }

    __host__ __device__
    inline
    array_size_t get_bytes() const
    {
        return elemsAlign_ * sizeof(T);
    }

    using strides_type = array_size_t[Dims - 1];
    __host__ __device__
    inline
    const strides_type &get_strides() const
    {
        return strides_;
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
        if (dim == Dims - 1) return sizeAlign_;
        else return this->sizes_[dim];
    }

    using sizes_type = std::array<array_size_t, Dims>;
    __host__
    inline
    sizes_type dims() const
    {
        sizes_type ret;
        for (auto dim : utils::make_range(Dims)) {
            ret[dim] = this->dim(dim);
        }
        return ret;
    }

    __host__
    inline
    sizes_type dims_align() const
    {
        sizes_type ret;
        for (auto dim : utils::make_range(Dims)) {
            ret[dim] = this->dim_align(dim);
        }
        return ret;
    }

    CUDARRAYS_TESTED(storage_test, dim_manager)
    CUDARRAYS_TESTED(storage_test, dim_manager_get_dim)
};

}

#endif
