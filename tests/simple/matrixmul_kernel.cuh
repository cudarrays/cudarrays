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

#ifndef _KERNEL_MATRIXMUL_H_
#define _KERNEL_MATRIXMUL_H_

#include <cudarrays/types.hpp>
#include <cudarrays/gpu.cuh>

static const size_t         MATRIXMUL_TILE_N = 16;
static const size_t MATRIXMUL_TILE_TB_HEIGHT = 16;

using namespace cudarrays;

template <typename StorageC, typename StorageA, typename StorageB>
__global__
void
matrixmul_kernel( matrix_view<float, layout::cmo, noalign, StorageC> C,
                 matrix_cview<float, layout::cmo, noalign, StorageA> A,
                 matrix_cview<float, layout::rmo, noalign, StorageB> B)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Partial results
    float partial[MATRIXMUL_TILE_N];
    for (int i = 0; i < MATRIXMUL_TILE_N; i++) partial[i] = 0.0f;

    // Flattened thread id
    int mid = ty * blockDim.x + tx;
    // RowId A/C (bx * TILE_N * TB_HEIGHT) + ty * TILE_N + tx
    int row = bx * (MATRIXMUL_TILE_N * MATRIXMUL_TILE_TB_HEIGHT) + mid;
    // ColId B
    int col = by * MATRIXMUL_TILE_N + tx;

    __shared__ float b_s[MATRIXMUL_TILE_TB_HEIGHT][MATRIXMUL_TILE_N];

    int iter = A.dim(1);

    // Compute tiles
    for (int i = 0; i < iter; i += MATRIXMUL_TILE_TB_HEIGHT) {
        // Load a 2D tile from B
        b_s[ty][tx] = B(i + ty, col);
        __syncthreads();

        for (int j = 0; j < MATRIXMUL_TILE_TB_HEIGHT; ++j) {
            // Load sub-row from A with different strides for each iteration
            float a = A(row, i + j);

            // Use value from A to compute
            for (int kk = 0; kk < MATRIXMUL_TILE_N; ++kk) {
                partial[kk] += a * b_s[j][kk];
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < MATRIXMUL_TILE_N; i++) {
        C(row, i + by * MATRIXMUL_TILE_N) = partial[i];
    }
}

#endif

/* vim:set ft=cuda backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
