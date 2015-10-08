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
#ifndef CUDARRAYS_SYSTEM_HPP_
#define CUDARRAYS_SYSTEM_HPP_

#include "common.hpp"
#include "utils.hpp"

namespace cudarrays {

extern std::vector<cudaStream_t> StreamsIn;
extern std::vector<cudaStream_t> StreamsOut;
extern std::vector<cudaEvent_t> EventsBegin;
extern std::vector<cudaEvent_t> EventsEnd;

namespace system {

extern utils::option<unsigned> MAX_GPUS;
extern utils::option<array_size_t> CUDA_VM_ALIGN;

extern unsigned GPUS;
extern unsigned PEER_GPUS;

template <typename T>
static inline array_size_t
vm_cuda_align_elems()
{
    // LIBRARY ENTRY POINT
    cudarrays_entry_point();

    return CUDA_VM_ALIGN/sizeof(T);
}

/**
 * Obtain the number of GPUs in the system
 * @return The number of GPUs in the system
 */
unsigned gpu_count();

/**
 * Obtain the number of GPUs in the system with P2P access support
 * @return The number of GPUs in the system with P2P access support
 */
unsigned peer_gpu_count();

void init();

} // namespace system
} // namespace cudarrays

#endif
