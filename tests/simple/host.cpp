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

#include <array>
#include <iostream>

#include <cudarrays/common.hpp>

#include <cudarrays/types.hpp>
#include <cudarrays/launch.hpp>

using namespace cudarrays;

unsigned INPUTSET = 0;
bool TEST = true;

array_size_t VECADD_ELEMS[1] = { 1024L * 1024L };

bool
launch_test_vecadd()
{
    static const array_size_t ELEMS = VECADD_ELEMS[INPUTSET];

    using my_array = vector<float>;

    my_array A{{ELEMS}};
    my_array B{{ELEMS}};
    my_array C{{ELEMS}};

    {
        for (unsigned i = 0; i < ELEMS; ++i) {
            A(i)      = float(i);
            B(i)      = float(i + 1.f);

            C(i)      = 0.f;
        }
    }

    {
        for (unsigned i = 0; i < ELEMS; ++i) {
            C(i) = A(i) + B(i);
        }
    }

    return true;
}

int main()
{
    launch_test_vecadd();

    return 0;
}

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
