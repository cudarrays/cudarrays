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

#include <memory>

#include "compiler.hpp"
#include "config.hpp"
#include "utils.hpp"

namespace cudarrays {

class host_storage {
private:
    struct state {
        void *data_          = nullptr;
        array_size_t offset_ = 0;
        size_t hostSize_     = 0;
    };

    // Store the state of the object in the heap to minimize the size in the GPU
    std::unique_ptr<state> state_;

private:
    void free_data();

public:
    host_storage();

    virtual ~host_storage();

    void alloc(array_size_t bytes, array_size_t offset, void *addr = nullptr);

    template <typename T = void>
    inline const T *
    addr() const
    {
        return reinterpret_cast<const T *>(reinterpret_cast<const T *>(state_->data_));
    }

    template <typename T = void>
    inline T *
    addr()
    {
        return reinterpret_cast<T *>(reinterpret_cast<T *>(state_->data_));
    }

    template <typename T = void>
    inline const T *
    base_addr() const
    {
        return reinterpret_cast<const T *>(reinterpret_cast<const char *>(state_->data_) - state_->offset_);
    }

    template <typename T = void>
    inline T *
    base_addr()
    {
        return reinterpret_cast<T *>(reinterpret_cast<char *>(state_->data_) - state_->offset_);
    }

    inline size_t
    size() const
    {
        return state_->hostSize_;
    }

    CUDARRAYS_TESTED(storage_test, host_storage)
};

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
