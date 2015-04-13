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
#ifndef CUDARRAYS_COMPILER_
#define CUDARRAYS_COMPILER_

#include <stdint.h>

#include <map>
#include <set>
#include <vector>

#include <cuda_runtime_api.h>

#if __CUDA_ARCH__ == 0
#undef __CUDA_ARCH__
#endif

#ifdef __NVCC__
    #define __hostdevice__ __device__ __host__
    #define __kernel__ __global__

#ifdef CUDARRAYS_LLVM
    #define __array_index__  __host__ __device__ __noinline__
    #define __array_bounds__ __host__ __device__ __noinline__
#else // CUDARRAYS_LLVM
    #define __array_index__  __host__ __device__ __forceinline__
    #define __array_bounds__ __host__ __device__ __forceinline__
#endif

#else // __NVCC__
    #define __hostdevice__
    #ifndef __host__
        #define __host__
    #endif
    #ifndef __device__
        #define __device__
    #endif
    #define __kernel__
#ifdef CUDARRAYS_LLVM
    #ifndef __array_index__
        #define __array_index__ __attribute__((noinline))
    #endif
    #ifndef __array_bounds__
        #define __array_bounds__ __attribute__((noinline))
    #endif
#else // CUDARRAYS_LLVM
    #ifndef __array_index__
        #define __array_index__ inline
    #endif
    #ifndef __array_bounds__
        #define __array_bounds__ inline
    #endif
#endif
#endif

// Compiler API to obtain information about array access patterns
extern "C" {
void
cudarrays_compiler_reset_info(const void *fun);

void
cudarrays_compiler_set_array_info(const void *fun, unsigned arrayArgIdx, unsigned ndims, uint8_t isRead, uint8_t isWritten);

void
cudarrays_compiler_set_array_dim_info(const void *fun, unsigned arrayArgIdx, unsigned arrayDim, unsigned gridDim);

extern bool CUDARRAYS_COMPILER_INFO;

void
cudarrays_compiler_register_info__();
}

namespace cudarrays {

struct compiler_dim_info {
    std::set<unsigned> gridDims_;
};

struct compiler_array_info {
    unsigned ndims_;
    bool isRead_;
    bool isWrite_;
    std::vector<compiler_dim_info> dimsInfo_;

    compiler_array_info(unsigned ndims, bool isRead, bool isWrite) :
        ndims_(ndims),
        isRead_(isRead),
        isWrite_(isWrite),
        dimsInfo_(ndims)
    {
    }
};

const compiler_array_info *
compiler_get_array_info(const void *fun, unsigned arrayArgIdx);

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
