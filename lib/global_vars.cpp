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
#include <unistd.h>
#include <vector>

#include <cuda_runtime_api.h>

#include "cudarrays/common.hpp"
#include "cudarrays/memory.hpp"
#include "cudarrays/storage.hpp"

namespace cudarrays {

std::vector<cudaStream_t> StreamsIn;
std::vector<cudaStream_t> StreamsOut;
std::vector<cudaEvent_t> EventsBegin;
std::vector<cudaEvent_t> EventsEnd;

utils::option<bool> LOG_DEBUG{"CUDARRAYS_LOG_DEBUG", false};
utils::option<bool> LOG_TRACE{"CUDARRAYS_LOG_TRACE", false};
utils::option<bool> LOG_VERBOSE{"CUDARRAYS_LOG_VERBOSE", false};
utils::option<bool> LOG_SHOW_PATH{"CUDARRAYS_LOG_SHOW_PATH", false};
utils::option<bool> LOG_SHORT_PATH{"CUDARRAYS_LOG_SHORT_PATH", false};
utils::option<bool> LOG_SHOW_SYMBOL{"CUDARRAYS_LOG_SHOW_SYMBOL", false};
utils::option<bool> LOG_STRIP_NAMESPACE{"CUDARRAYS_LOG_STRIP_NAMESPACE", false};

#if defined(CUDARRAYS_TRACE_BLOCK) || defined(CUDARRAYS_TRACE_WARP)
__attribute__((weak))
void *trace_start_buffer__ = NULL;
__attribute__((weak))
void *trace_end_buffer__ = NULL;
#endif
}

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
