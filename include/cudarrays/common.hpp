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
#ifndef CUDARRAYS_COMMON_HPP_
#define CUDARRAYS_COMMON_HPP_

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_functions.h>

#include <functional>

#include <cassert>
#include <cstdlib>

#include <cstdint>

#include "utils.hpp"

using namespace utils;

#include "compiler.hpp"
#include "log.hpp"

namespace cudarrays {

#ifdef LONG_INDEX
using array_index_t = int64_t;
using array_size_t = uint64_t;
#else
using array_index_t = int32_t;
using array_size_t = uint32_t;
#endif

extern unsigned MAX_GPUS;
extern unsigned PEER_GPUS;

extern array_size_t CUDA_VM_ALIGN;

template <typename T>
static inline array_size_t
CUDA_VM_ALIGN_ELEMS()
{
    return CUDA_VM_ALIGN/sizeof(T);
}

using handler_fn = std::function<bool (bool)>;

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

void init_lib();
void fini_lib();

#ifdef CUDARRAYS_UNITTEST
#define CUDARRAYS_TESTED(C,T) friend ::C##_##T##_Test;
#else
#define CUDARRAYS_TESTED(C,T)
#endif

#define CUDA_CALL(x)                                       \
do {                                                       \
    cudaError_t err__ = (x);                               \
    if (err__ != cudaSuccess) {                            \
        fprintf(stderr,                                    \
                "Error calling CUDA: %d. Message: '%s'\n", \
                err__,                                     \
                cudaGetErrorString(err__));                \
        abort();                                           \
    }                                                      \
} while (0)

}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
