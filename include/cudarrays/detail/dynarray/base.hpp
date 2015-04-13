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
};

template <typename T, unsigned Dims>
class dim_manager {
public:
    static constexpr unsigned FirstDim = 3 - Dims;

    static constexpr unsigned DimIdxZ = 0 - FirstDim;
    static constexpr unsigned DimIdxY = 1 - FirstDim;
    static constexpr unsigned DimIdxX = 2 - FirstDim;

    array_size_t sizes_[Dims];
    array_size_t *sizesAlign_;
    array_size_t offsAlign_[Dims - 1];

private:
    array_size_t elems_;
    array_size_t offset_;

    array_size_t elemsAlign_;

    __host__ void
    init(const extents<Dims> &extents,
         align_t align,
         const std::array<array_size_t, Dims - 1> &offAlignImpl)
    {
        // Initialize array sizes
        utils::copy(extents, sizes_);
        std::copy(extents.begin(), extents.end(), sizesAlign_);

        // Check if the implementation imposes a bigger alignment than the user
        if (Dims > 1 && offAlignImpl[Dims - 2] > align.alignment) {
            ASSERT(offAlignImpl[Dims - 2] % align.alignment == 0);
            align.alignment = offAlignImpl[Dims - 2];
        }

        // Compute offset and aligned size of the lowest order dimension
        offset_ = 0;
        if (align.alignment > 1) {
            if (align.position > align.alignment) {
                offset_ = utils::round_next(align.position, align.alignment) - align.position;
            } else if (align.position > 0) {
                offset_ = align.alignment - align.position;
            }
            sizesAlign_[Dims - 1] = utils::round_next(extents[Dims - 1] + offset_, align.alignment);
        } else {
            sizesAlign_[Dims - 1] = extents[Dims - 1];
        }

        // Fill offsets' array
        array_size_t nextOffAlign = 1;
        for (int i = Dims - 1; i > 0; --i) {
            nextOffAlign *= sizesAlign_[i];

            if (offAlignImpl[i - 1] > 0) {
                nextOffAlign = utils::round_next(nextOffAlign, offAlignImpl[i - 1]);
            }

            offsAlign_[i - 1] = nextOffAlign;
        }

        // Compute number of elements
        elems_      = std::accumulate(sizes_, sizes_ + Dims,
                                      1, std::multiplies<array_size_t>());
        elemsAlign_ = std::accumulate(sizesAlign_, sizesAlign_ + Dims,
                                      1, std::multiplies<array_size_t>());
    }

public:
    __host__
    dim_manager(extents<Dims> extents,
                const align_t &align,
                const std::array<array_size_t, Dims - 1> &offAlignImpl = std::array<array_size_t, Dims - 1>())
    {
        ASSERT(extents.size() == Dims);

        sizesAlign_ = new array_size_t[Dims];
        init(extents, align, offAlignImpl);
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
    array_size_t get_offset() const
    {
        return offset_;
    }

    __host__ __device__
    inline
    void set_offset(array_size_t offset)
    {
        offset_ = offset;
    }

    __host__ __device__
    inline
    array_size_t get_dim(unsigned dim) const
    {
        return this->sizes_[dim];
    }

    __host__ __device__
    inline
    array_size_t get_dim_align(unsigned dim) const
    {
        return this->sizesAlign_[dim];
    }

    using sizes_type = array_size_t[Dims];
    __host__ __device__
    inline
    const sizes_type &get_sizes() const
    {
        return sizes_;
    }

    CUDARRAYS_TESTED(storage_test, dim_manager)
    CUDARRAYS_TESTED(storage_test, dim_manager_get_dim)
};

template <typename T>
class host_storage {
private:
    struct state {
        T * data_;
        array_size_t offset_;
        size_t hostSize_;
    };

    // Store the state of the object in the heap to minimize the size in the GPU
    state *state_;

private:
    void free_data()
    {
        state_->data_ -= state_->offset_;

        int ret = munmap(state_->data_, state_->hostSize_);
        ASSERT(ret == 0);
        state_->data_ = nullptr;
    }

public:
    __host__
    host_storage()
    {
        state_ = new state;
        state_->data_ = nullptr;
    }

    __host__
    virtual ~host_storage()
    {
        if (state_->data_ != nullptr) {
            free_data();
        }
        delete state_;
        state_ = nullptr;
    }

    void alloc(array_size_t elems, array_size_t offset, T *addr = nullptr)
    {
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
        if (addr != nullptr) flags |= MAP_FIXED;
        state_->hostSize_ = size_t(elems) * sizeof(T);
        state_->data_ = (T *) mmap(addr, state_->hostSize_,
                                   PROT_READ | PROT_WRITE,
                                   flags, -1, 0);

        if (addr != nullptr && state_->data_ != addr) {
            FATAL("%p vs %p", state_->data_, addr);
        }
        DEBUG("mmapped: %p (%zd)", state_->data_, state_->hostSize_);

        state_->data_  += offset;
        state_->offset_ = offset;
    }

    const T *
    get_addr() const
    {
        return state_->data_;
    }

    T *
    get_addr()
    {
        return state_->data_;
    }

    const T *
    get_base_addr() const
    {
        return state_->data_ - state_->offset_;
    }

    T *
    get_base_addr()
    {
        return state_->data_ - state_->offset_;
    }

    size_t
    size() const
    {
        return state_->hostSize_;
    }

    CUDARRAYS_TESTED(storage_test, host_storage)
};

template <typename T, unsigned Dims>
class dynarray_base {
public:
    using  dim_manager_type = dim_manager<T, Dims>;
    using host_storage_type = host_storage<T>;

    dynarray_base(extents<Dims> extents,
                  const align_t &align) :
        dimManager_(extents, align)
    {
        // Alloc host memory
        hostStorage_.alloc(this->get_dim_manager().get_elems_align(), this->get_dim_manager().get_offset());
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

    inline __host__ __device__
    const host_storage_type &
    get_host_storage() const
    {
        return hostStorage_;
    }

    inline __host__ __device__
    host_storage_type &
    get_host_storage()
    {
        return hostStorage_;
    }

    virtual void set_current_gpu(unsigned /*idx*/) {}
    virtual void to_device() = 0;
    virtual void to_host() = 0;

private:
    dim_manager_type dimManager_;
    host_storage_type hostStorage_;
};

template <typename T, unsigned Dims, storage_impl StorageImpl, typename PartConf>
class dynarray_storage;

}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
