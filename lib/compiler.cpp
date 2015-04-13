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
#include <cstdio>


#include "cudarrays/compiler.hpp"

bool CUDARRAYS_COMPILER_INFO = true;

typedef std::map<unsigned, cudarrays::compiler_array_info> function_info;

typedef std::map<const void *, function_info> function_map;

function_map functionsInfo;

void
cudarrays_compiler_reset_info(const void *fun)
{

    fprintf(stderr, "%p\n", fun);

    function_map::iterator it = functionsInfo.find(fun);
    if (it != functionsInfo.end()) {
        it->second.clear();
    } else {
        functionsInfo[fun] = function_info();
    }
}

void
cudarrays_compiler_set_array_info(const void *fun, unsigned arrayArgIdx, unsigned ndims, uint8_t _isRead, uint8_t _isWritten)
{
    bool isRead    = bool(_isRead);
    bool isWritten = bool(_isWritten);

    fprintf(stderr, "%p: Idx: %u Dims: %u R: %d W: %d\n", fun, arrayArgIdx, ndims, isRead, isWritten);

    function_map::iterator it = functionsInfo.find(fun);
    if (it == functionsInfo.end()) {
        abort();
    }
    function_info::iterator it2 = it->second.find(arrayArgIdx);
    if (it2 == it->second.end()) {
        cudarrays::compiler_array_info info(ndims, isRead, isWritten);
        it2 = it->second.insert(std::make_pair(arrayArgIdx, info)).first;
    } else {
        abort();
    }
}

void
cudarrays_compiler_set_array_dim_info(const void *fun, unsigned arrayArgIdx, unsigned arrayDim, unsigned gridDim)
{

    fprintf(stderr, "%p: Idx: %u A: %u G: %u\n", fun, arrayArgIdx, arrayDim, gridDim);

    function_map::iterator it = functionsInfo.find(fun);
    if (it == functionsInfo.end()) {
        abort();
    }
    function_info::iterator it2 = it->second.find(arrayArgIdx);
    if (it2 == it->second.end()) {
        abort();
    }
    it2->second.dimsInfo_[arrayDim].gridDims_.insert(gridDim);
}


void
cudarrays_compiler_register_info__()
{
    CUDARRAYS_COMPILER_INFO = false;
}

namespace cudarrays {

const compiler_array_info *
compiler_get_array_info(const void *fun, unsigned arrayArgIdx)
{
    function_map::iterator it = functionsInfo.find(fun);
    if (it == functionsInfo.end()) {
        return NULL;
    }
    function_info::iterator it2 = it->second.find(arrayArgIdx);
    if (it2 == it->second.end()) {
        return NULL;
    }
    return &it2->second;
}

}

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
