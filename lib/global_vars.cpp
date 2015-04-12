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

#include <cstdlib>
#include <cstddef>

#include "cudarrays/common.hpp"
#include "cudarrays/storage.hpp"
#include "cudarrays/memory.hpp"

namespace cudarrays {

bool OPTION_DEBUG;

unsigned MAX_GPUS;
unsigned PEER_GPUS;

array_size_t CUDA_VM_ALIGN;

array_size_t PAGE_ALIGN;
array_size_t PAGES_PER_ARENA;

#if defined(CUDARRAYS_TRACE_BLOCK) || defined(CUDARRAYS_TRACE_WARP)
__attribute__((weak))
void *trace_start_buffer__ = NULL;
__attribute__((weak))
void *trace_end_buffer__ = NULL;
#endif
}
