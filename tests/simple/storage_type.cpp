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

#include <iostream>

#include <cudarrays/dynarray_view.hpp>

using namespace cudarrays;

template <typename Layout>
using my_array = dynarray<float ***, Layout, replicate::none>;

int main()
{
    static const array_size_t X = 2;
    static const array_size_t Y = 3;
    static const array_size_t Z = 4;

    static const array_size_t elems = X * Y * Z;

    auto A_rmo = make_array<float ***, layout::rmo>({ Z, Y, X });
    auto A_cmo = make_array<float ***, layout::cmo>({ Z, Y, X });

    float c = 0.f;
    for (unsigned i = 0; i < Z; ++i) {
        for (unsigned j = 0; j < Y; ++j) {
            for (unsigned k = 0; k < X; ++k) {
                A_rmo(i, j, k) = c;
                A_cmo(i, j, k) = c;

                c += 1.f;
            }
        }
    }

    const float *P_rmo = (const float *)A_rmo.host_addr();
    const float *P_cmo = (const float *)A_cmo.host_addr();

    for (unsigned i = 0; i < elems; ++i) {
        printf("%f vs %f\n", P_rmo[i], P_cmo[i]);
    }

    return 0;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
