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

#include <sys/mman.h>

#include "common.hpp"

namespace cudarrays {

template <typename StorageTraits>
class host_storage {
public:
    using     value_type = typename StorageTraits::value_type;
    using alignment_type = typename StorageTraits::alignment_type;

    host_storage() :
        data_{nullptr}
    {
    }

    virtual ~host_storage()
    {
        if (data_ != nullptr) {
            int ret = munmap(this->base_addr(), hostSize_);
            ASSERT(ret == 0);
            data_ = nullptr;
        }
    }

    void alloc(array_size_t bytes, value_type *addr = nullptr)
    {
        int flags = MAP_PRIVATE | MAP_ANONYMOUS;
        if (addr != nullptr) flags |= MAP_FIXED;
        hostSize_ = bytes;
        data_ = (value_type *) mmap(addr, hostSize_,
                                    PROT_READ | PROT_WRITE,
                                    flags, -1, 0);

        if (addr != nullptr && data_ != addr) {
            FATAL("Unable to map memory at %p", addr);
        }
        DEBUG("host> mmapped: %p (%zd)", data_, hostSize_);

        data_ = data_ + alignment_type::get_offset();
    }

    inline const value_type *
    addr() const
    {
        return data_;
    }

    inline value_type *
    addr()
    {
        return data_;
    }

    inline const value_type *
    base_addr() const __attribute__((assume_aligned(4096)))
    {
        return data_ - alignment_type::get_offset();
    }

    inline value_type *
    base_addr() __attribute__((assume_aligned(4096)))
    {
        return data_ - alignment_type::get_offset();
    }

    inline size_t
    size() const
    {
        return hostSize_;
    }

private:
    value_type *data_ = nullptr;
    size_t hostSize_  = 0;

    CUDARRAYS_TESTED(storage_test, host_storage)
};

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
