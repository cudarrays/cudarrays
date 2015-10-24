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
#include <cudarrays/launch.hpp>

#include "common.hpp"

using namespace cudarrays;

template <typename StorageImpl>
using float_array = dynarray<float ***, layout::rmo, StorageImpl>;

int main()
{
    static const int NONE = -1;

    auto A_x_x = make_array<float ***, layout::rmo, noalign, automatic::none>({ 1500, 300, 600 });
    auto A_x_y = make_array<float ***, layout::rmo, noalign, automatic::none>({ 1500, 300, 600 });
    A_x_x.distribute<3>({{compute::x, 6}, {NONE, NONE, 0}});
    A_x_y.distribute<3>({{compute::y, 6}, {NONE, NONE, 1}});

    auto A_y_x = make_array<float ***, layout::rmo, noalign, reshape::y>({ 1500, 300, 600 });
    auto A_y_y = make_array<float ***, layout::rmo, noalign, reshape::y>({ 1500, 300, 600 });
    A_y_x.distribute<3>({{compute::x, 6}, {NONE, 0, NONE}});
    A_y_y.distribute<3>({{compute::y, 6}, {NONE, 1, NONE}});

    auto A_xy1 = make_array<float ***, layout::rmo, noalign, reshape::xy>({ 1500, 300, 600 });
    auto A_xy2 = make_array<float ***, layout::rmo, noalign, reshape::xy>({ 1500, 300, 600 });
    A_xy1.distribute<3>({{compute::xy, 6}, {NONE, 1, 0}});
    A_xy2.distribute<3>({{compute::xy, 6}, {NONE, 0, 1}});

    auto A_xz1 = make_array<float ***, layout::rmo, noalign, reshape::xz>({ 1500, 300, 600 });
    auto A_xz2 = make_array<float ***, layout::rmo, noalign, reshape::xz>({ 1500, 300, 600 });
    A_xz1.distribute<3>({{compute::xz, 6}, {2, NONE, 0}});
    A_xz2.distribute<3>({{compute::xz, 6}, {0, NONE, 2}});

    auto A_yz1 = make_array<float ***, layout::rmo, noalign, reshape::yz>({ 1500, 300, 600 });
    auto A_yz2 = make_array<float ***, layout::rmo, noalign, reshape::yz>({ 1500, 300, 600 });
    A_yz1.distribute<3>({{compute::yz, 6}, {2, 1, NONE}});
    A_yz2.distribute<3>({{compute::yz, 6}, {1, 2, NONE}});

    return 0;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
