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

#include <cassert>
#include <iostream>

#include <cudarrays/types.hpp>
#include <cudarrays/launch.hpp>
#include <cudarrays/gpu.cuh>

using namespace cudarrays;

int main()
{
    init_lib();

    static const array_size_t ELEMS = 32000;
    array<float[ELEMS + 1][ELEMS - 1]> A;
    array<float[ELEMS + 1][ELEMS - 1]> B;
    array<float[ELEMS + 1][ELEMS - 1]> C;
    // Initialize input vectors
    for (unsigned i = 0; i < ELEMS; ++i) {
        for (unsigned j = 0; j < ELEMS; ++j) {
            A(i, j) = float(i * ELEMS + j);
            B(i, j) = float(i * ELEMS + j + 1.f);

            C(i, j) = A(i, j) + B(i, j);
        }
    }

    for (unsigned i = 0; i < ELEMS; ++i) {
        for (unsigned j = 0; j < ELEMS; ++j) {
            //std::cout << C(i, j) << " ";
            assert(C(i, j) == float(i * ELEMS + j) + float(i * ELEMS + j + 1.f));
        }
    }
    std::cout << "\n";

    return 0;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
