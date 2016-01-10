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
#ifndef CUDARRAYS_COHERENCE_HPP_
#define CUDARRAYS_COHERENCE_HPP_

#include <vector>

#include "detail/utils/base.hpp"

namespace cudarrays {

class coherent;

class coherence_policy {
public:
    virtual ~coherence_policy() noexcept {}
    virtual void release(const std::vector<unsigned> &gpus, bool Const) = 0;
    virtual void acquire() = 0;

    virtual void bind(coherent &obj) = 0;
    virtual void unbind() = 0;
};

class coherent :
    public base<coherent> {
public:
    virtual ~coherent() noexcept {}
    virtual coherence_policy &get_coherence_policy() noexcept = 0;
    virtual void set_current_gpu(unsigned /*idx*/) = 0;

    virtual bool is_distributed() const = 0;
    virtual bool distribute(const std::vector<unsigned> &gpus) = 0;

    virtual void to_device() = 0;
    virtual void to_host() = 0;

    virtual void *host_addr() noexcept = 0;
    virtual const void *host_addr() const noexcept = 0;

    virtual size_t size() const noexcept = 0;
};

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
