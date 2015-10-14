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

#ifndef _KERNEL_STENCIL_H_
#define _KERNEL_STENCIL_H_

#include <cudarrays/types.hpp>
#include <cudarrays/gpu.cuh>

using namespace cudarrays;

static const int STENCIL = 4;
static const int STENCIL_BLOCK_X = 8;
static const int STENCIL_BLOCK_Y = 8;

template <typename StorageB, typename StorageA>
__global__
void
stencil_kernel( matrix_view<float, layout::rmo, noalign, StorageB> B,
               matrix_cview<float, layout::rmo, noalign, StorageA> A)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int sx = tx + STENCIL;
    int sy = ty + STENCIL;

    int j = tx + bx * blockDim.x + STENCIL;
    int i = ty + by * blockDim.y + STENCIL;

    float val = A(i, j);

    __shared__ float tile[STENCIL_BLOCK_Y + 2 * STENCIL][STENCIL_BLOCK_X + 2 * STENCIL];

    tile[sy][sx] = val;

    if (tx < STENCIL) {
        tile[sy][tx                       ] = A(i, j - STENCIL);
        tile[sy][tx + blockDim.x + STENCIL] = A(i, j + blockDim.x);
    }
    if (ty < STENCIL) {
        tile[ty                       ][sx] = A(i - STENCIL, j);
        tile[ty + blockDim.y + STENCIL][sx] = A(i + blockDim.y, j);
    }

    __syncthreads();

    for (int k = 1; k <= STENCIL; ++k) {
        val += 3.f * (tile[sy][sx - k] + tile[sy][sx + k]) +
               2.f * (tile[sy - k][sx] + tile[sy + k][sx]);
    }

    B(i, j) = val;
}

#endif

/* vim:set ft=cuda backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
