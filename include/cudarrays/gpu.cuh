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
#ifndef CUDARRAYS_GPU_CUH_
#define CUDARRAYS_GPU_CUH_

#include <cuda_runtime_api.h>

#include "common.hpp"

namespace cudarrays {

#ifdef CUDARRAYS_DONT_OVERRIDE_BLOCK_IDX

#define blockIdx     blockIdx
#define blockIdx_yx  dim3(blockIdx.y, blockIdx.x)
#define blockIdx_yxz dim3(blockIdx.y, blockIdx.x, blockIdx.z)
#define blockIdx_yzx dim3(blockIdx.y, blockIdx.z, blockIdx.x)
#define blockIdx_zxy dim3(blockIdx.z, blockIdx.x, blockIdx.y)
#define blockIdx_zyx dim3(blockIdx.z, blockIdx.y, blockIdx.x)

#define gridDim      gridDim
#define gridDim_yx   dim3(gridDim.y, gridDim.x)
#define gridDim_yxz  dim3(gridDim.y, gridDim.x, gridDim.z)
#define gridDim_yzx  dim3(gridDim.y, gridDim.z, gridDim.x)
#define gridDim_zxy  dim3(gridDim.z, gridDim.x, gridDim.y)
#define gridDim_zyx  dim3(gridDim.z, gridDim.y, gridDim.x)

#else

#define blockIdx     dim3(blockIdx.x + global_offset_x, blockIdx.y + global_offset_y, blockIdx.z + global_offset_z)
#define blockIdx_yx  dim3(blockIdx.y + global_offset_y, blockIdx.x + global_offset_x)
#define blockIdx_yxz dim3(blockIdx.y + global_offset_y, blockIdx.x + global_offset_x, blockIdx.z + global_offset_z)
#define blockIdx_yzx dim3(blockIdx.y + global_offset_y, blockIdx.z + global_offset_z, blockIdx.x + global_offset_x)
#define blockIdx_zxy dim3(blockIdx.z + global_offset_z, blockIdx.x + global_offset_x, blockIdx.y + global_offset_y)
#define blockIdx_zyx dim3(blockIdx.z + global_offset_z, blockIdx.y + global_offset_y, blockIdx.x + global_offset_x)

#define gridDim      dim3(global_grid_x, global_grid_y, global_grid_z)
#define gridDim_yx   dim3(global_grid_y, global_grid_x)
#define gridDim_yxz  dim3(global_grid_y, global_grid_x, global_grid_z)
#define gridDim_yzx  dim3(global_grid_y, global_grid_z, global_grid_x)
#define gridDim_zxy  dim3(global_grid_z, global_grid_x, global_grid_y)
#define gridDim_zyx  dim3(global_grid_z, global_grid_y, global_grid_x)

#endif

extern __constant__
unsigned global_grid_x;
extern __constant__
unsigned global_grid_y;
extern __constant__
unsigned global_grid_z;
extern __constant__
unsigned global_offset_x;
extern __constant__
unsigned global_offset_y;
extern __constant__
unsigned global_offset_z;

#if defined(CUDARRAYS_TRACE_BLOCK) || defined(CUDARRAYS_TRACE_WARP)
extern __constant__
unsigned long long *tracer_tstamps_start;
extern __constant__
unsigned long long *tracer_tstamps_end;
#endif

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
