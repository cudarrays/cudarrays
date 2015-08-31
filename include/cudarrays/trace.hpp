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
#ifndef CUDARRAYS_TRACE_HPP_
#define CUDARRAYS_TRACE_HPP_

#include <cuda_runtime_api.h>

#include <cassert>

#define WARP_SHIFT 5

namespace cudarrays {

extern void *trace_start_buffer__;
extern void *trace_end_buffer__;

enum TracerGranularity {
    WARP  = 1,
    BLOCK = 2
};

#if defined(CUDARRAYS_TRACE_BLOCK) || defined(CUDARRAYS_TRACE_WARP)

template <unsigned Dims, TracerGranularity Granularity, bool Inv = false>
class trace_gpu;

#ifdef __CUDA_ARCH__

template <unsigned Dims, bool Inv>
struct compute_cta_id;

template <>
struct compute_cta_id<1, false>
{
    __device__
    static inline unsigned execute()
    {
        return blockIdx.x;
    }
};

template <>
struct compute_cta_id<2, false>
{
    __device__
    static inline unsigned execute()
    {
        return blockIdx.y * gridDim.x +
               blockIdx.x;
    }
};

template <>
struct compute_cta_id<3, false>
{
    __device__
    static inline unsigned execute()
    {
        return blockIdx.z * gridDim.x * gridDim.y +
               blockIdx.y * gridDim.x              +
               blockIdx.x;
    }
};

template <>
struct compute_cta_id<1, true>
{
    __device__
    static inline unsigned execute()
    {
        return blockIdx_inv.x;
    }
};

template <>
struct compute_cta_id<2, true>
{
    __device__
    static inline unsigned execute()
    {
        return blockIdx.y * blockDim.x +
               blockIdx.x;
    }
};

template <>
struct compute_cta_id<3, true>
{
    __device__
    static inline unsigned execute()
    {
        return blockIdx.z * blockDim.x * blockDim.y +
               blockIdx.y * blockDim.x              +
               blockIdx.x;
    }
};


#endif

template <unsigned Dims, bool Inv>
class trace_gpu<Dims, WARP, Inv>
{
#ifdef __CUDA_ARCH__
public:
    inline __device__
    trace_gpu()
    {
        if ((threadIdx.x & (32 - 1)) == 0) {
            unsigned cta_id = compute_cta_id<Dims, Inv>::execute();
            unsigned warp_id;
            unsigned warps;
            asm("{                         \n\
                    mov.u32 %0, %%warpid;  \n\
                    mov.u32 %1, %%nwarpid; \n\
                 }" : "=r"(warp_id), "=r"(warps));

            unsigned long long tstamp = clock64();
            ::tracer_tstamps_start[cta_id * warps + warp_id] = tstamp;
        }
    }

    inline __device__
    ~trace_gpu()
    {
        if ((threadIdx.x & (32 - 1)) == 0) {
            unsigned cta_id = compute_cta_id<Dims, Inv>::execute();
            unsigned warp_id;
            unsigned warps;
            asm("{                         \n\
                    mov.u32 %0, %%warpid;  \n\
                    mov.u32 %1, %%nwarpid; \n\
                 }" : "=r"(warp_id), "=r"(warps));

            unsigned long long tstamp = clock64();
            ::tracer_tstamps_end[cta_id * warps + warp_id] = tstamp;
        }
    }
#endif
};

template <unsigned Dims, bool Inv>
class trace_gpu<Dims, BLOCK, Inv>
{
#ifdef __CUDA_ARCH__
public:
    inline __device__
    trace_gpu()
    {
        if (threadIdx.x == 0) {
            unsigned cta_id = compute_cta_id<Dims, Inv>::execute();
            unsigned long long tstamp = clock64();
            ::tracer_tstamps_start[cta_id] = tstamp;
        }
    }

    inline __device__
    ~trace_gpu()
    {
        if (threadIdx.x == 0) {
            unsigned cta_id = compute_cta_id<Dims, Inv>::execute();
            unsigned long long tstamp = clock64();
            tracer_tstamps_end[cta_id] = tstamp;
        }
    }
#endif
};

template <TracerGranularity Granularity>
class tracer;

template <>
class tracer<WARP>
{
public:
    __host__
    tracer(dim3 grid, dim3 block) :
        warps_((grid.x == 0? 1 : grid.x) *
               (grid.y == 0? 1 : grid.y) *
               (grid.z == 0? 1 : grid.z) *
               48),
        inCPU_(false)
    {
        traces_ = new unsigned long long[warps_ * 2];

        CUDA_CALL(cudaMalloc((void **)&tracesStartDev_, warps_ * sizeof(long long)));
        CUDA_CALL(cudaMalloc((void **)&tracesEndDev_, warps_ * sizeof(long long)));

        CUDA_CALL(cudaMemset(tracesStartDev_, 0, warps_ * sizeof(long long)));
        CUDA_CALL(cudaMemset(tracesEndDev_, 0, warps_ * sizeof(long long)));

        CUDA_CALL(cudaMemcpyToSymbol(trace_start_buffer__, &tracesStartDev_, sizeof(void *), 0));
        CUDA_CALL(cudaMemcpyToSymbol(trace_end_buffer__, &tracesEndDev_, sizeof(void *), 0));
    }

    __host__
    ~tracer()
    {
        delete [] traces_;

        CUDA_CALL(cudaFree(tracesStartDev_));
        CUDA_CALL(cudaFree(tracesEndDev_));
    }

    __host__
    long long ntraces() const
    {
        return warps_;
    }

private:
    unsigned long long warps_;
    bool inCPU_;

    unsigned long long *traces_;
    unsigned long long *tracesStartDev_;
    unsigned long long *tracesEndDev_;

    friend std::ofstream &operator<<(std::ofstream &os, const tracer& t);
};

template <>
class tracer<BLOCK>
{
public:
    __host__
    tracer(dim3 grid, dim3) :
        blocks_((grid.x == 0? 1 : grid.x) *
                (grid.y == 0? 1 : grid.y) *
                (grid.z == 0? 1 : grid.z)),
        inCPU_(false)
    {
        traces_ = new unsigned long long[blocks_ * 2];

        CUDA_CALL(cudaMalloc((void **)&tracesStartDev_, blocks_ * sizeof(long long)));
        CUDA_CALL(cudaMalloc((void **)&tracesEndDev_, blocks_ * sizeof(long long)));

        CUDA_CALL(cudaMemset(tracesStartDev_, 0, blocks_ * sizeof(long long)));
        CUDA_CALL(cudaMemset(tracesEndDev_, 0, blocks_ * sizeof(long long)));

        CUDA_CALL(cudaMemcpyToSymbol(trace_start_buffer__, &tracesStartDev_, sizeof(void *), 0));
        CUDA_CALL(cudaMemcpyToSymbol(trace_end_buffer__, &tracesEndDev_, sizeof(void *), 0));
    }

    __host__
    ~tracer()
    {
        delete [] traces_;

        CUDA_CALL(cudaFree(tracesStartDev_));
        CUDA_CALL(cudaFree(tracesEndDev_));
    }

    __host__
    long long ntraces() const
    {
        return blocks_;
    }

private:
    unsigned long long blocks_;
    bool inCPU_;

    unsigned long long *traces_;
    unsigned long long *tracesStartDev_;
    unsigned long long *tracesEndDev_;

    friend std::ofstream &operator<<(std::ofstream &os, const tracer& t);
};

#endif
}


#if defined(CUDARRAYS_TRACE_BLOCK) && defined(CUDARRAYS_TRACE_WARP)
#error "Block and Warp tracing cannot be enabled at the same time"
#endif

#if defined(CUDARRAYS_TRACE_BLOCK)
#define TRACE_GPU(d) cudarrays::trace_gpu<d, BLOCK> tracer__;
#define TRACER_DRIVER cudarrays::tracer<BLOCK>
#endif
#if defined(CUDARRAYS_TRACE_WARP)
#define TRACE_GPU(d) cudarrays::trace_gpu<d, WARP> tracer__;
#define TRACER_DRIVER cudarrays::tracer<WARP>
#endif

#if !defined(CUDARRAYS_TRACE_BLOCK) && !defined(CUDARRAYS_TRACE_WARP)
#define TRACE_GPU(d)
#define TRACER_DRIVER
#else
#define CUDARRAYS_TRACE
#endif

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
