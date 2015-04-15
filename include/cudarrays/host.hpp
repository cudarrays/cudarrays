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
#ifndef CUDARRAYS_HOST_HPP_
#define CUDARRAYS_HOST_HPP_

#include "compiler.hpp"

namespace cudarrays {

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

}

#endif
