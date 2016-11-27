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

#include "cudarrays/gpu.cuh"

namespace cudarrays {

__constant__
unsigned global_grid_x;
__constant__
unsigned global_grid_y;
__constant__
unsigned global_grid_z;
__constant__
unsigned global_offset_x;
__constant__
unsigned global_offset_y;
__constant__
unsigned global_offset_z;

void update_gpu_global_grid(dim3 total_grid)
{
    cudaError_t err;
#ifndef __NVCC__
    err = cudaMemcpyToSymbol(&global_grid_x, &total_grid.x, sizeof(total_grid.x), 0);
    assert(err == cudaSuccess);
    err = cudaMemcpyToSymbol(&global_grid_y, &total_grid.y, sizeof(total_grid.y), 0);
    assert(err == cudaSuccess);
    err = cudaMemcpyToSymbol(&global_grid_z, &total_grid.z, sizeof(total_grid.z), 0);
    assert(err == cudaSuccess);

#else
    err = ::cudaMemcpyToSymbol(&global_grid_x, &total_grid.x, sizeof(total_grid.x), 0);
    assert(err == cudaSuccess);
    err = ::cudaMemcpyToSymbol(&global_grid_y, &total_grid.y, sizeof(total_grid.y), 0);
    assert(err == cudaSuccess);
    err = ::cudaMemcpyToSymbol(&global_grid_z, &total_grid.z, sizeof(total_grid.z), 0);
    assert(err == cudaSuccess);
#endif
}

void update_gpu_offset(dim3 off)
{
    cudaError_t err;
#ifndef __NVCC__
    err = cudaMemcpyToSymbol(&global_offset_x, &off.x, sizeof(off.x), 0);
    assert(err == cudaSuccess);
    err = cudaMemcpyToSymbol(&global_offset_y, &off.y, sizeof(off.y), 0);
    assert(err == cudaSuccess);
    err = cudaMemcpyToSymbol(&global_offset_z, &off.z, sizeof(off.z), 0);
    assert(err == cudaSuccess);
#else
    err = ::cudaMemcpyToSymbol(&global_offset_x, &off.x, sizeof(off.x), 0);
    assert(err == cudaSuccess);
    err = ::cudaMemcpyToSymbol(&global_offset_y, &off.y, sizeof(off.y), 0);
    assert(err == cudaSuccess);
    err = ::cudaMemcpyToSymbol(&global_offset_z, &off.z, sizeof(off.z), 0);
    assert(err == cudaSuccess);
#endif
}

#if defined(CUDARRAYS_TRACE_BLOCK) || defined(CUDARRAYS_TRACE_WARP)
__constant__
unsigned long long *tracer_tstamps_start;
__constant__
unsigned long long *tracer_tstamps_end;
#endif

}
