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

#include <cudarrays/dynarray.hpp>
#include <cudarrays/launch.hpp>

#include "common.hpp"

namespace cudarrays {
void init_lib();
}

using namespace cudarrays;

template <unsigned Dims, typename StorageImpl>
using float_array = dynarray<float, Dims, layout::rmo, StorageImpl>;

int main(int argc, char *argv[])
{
    init_lib();

    static const int NONE = -1;

    auto *A_x_x = new float_array<3, automatic::none>({ 1500, 300, 600 });
    auto *A_x_y = new float_array<3, automatic::none>({ 1500, 300, 600 });
    A_x_x->distribute<3>({{partition::x, 6}, {NONE, NONE, 0}});
    A_x_y->distribute<3>({{partition::y, 6}, {NONE, NONE, 1}});
    delete A_x_x; delete A_x_y;

    auto *A_y_x = new float_array<3, reshape::y>({ 1500, 300, 600 });
    auto *A_y_y = new float_array<3, reshape::y>({ 1500, 300, 600 });
    A_y_x->distribute<3>({{partition::x, 6}, {NONE, 0, NONE}});
    A_y_y->distribute<3>({{partition::y, 6}, {NONE, 1, NONE}});
    delete A_y_x; delete A_y_y;

    auto *A_xy1 = new float_array<3, reshape::xy>({ 1500, 300, 600 });
    auto *A_xy2 = new float_array<3, reshape::xy>({ 1500, 300, 600 });
    A_xy1->distribute<3>({{partition::xy, 6}, {NONE, 1, 0}});
    A_xy2->distribute<3>({{partition::xy, 6}, {NONE, 0, 1}});
    delete A_xy1; delete A_xy2;

    auto *A_xz1 = new float_array<3, reshape::xz>({ 1500, 300, 600 });
    auto *A_xz2 = new float_array<3, reshape::xz>({ 1500, 300, 600 });
    A_xz1->distribute<3>({{partition::xz, 6}, {2, NONE, 0}});
    A_xz2->distribute<3>({{partition::xz, 6}, {0, NONE, 2}});
    delete A_xz1; delete A_xz2;

    auto *A_yz1 = new float_array<3, reshape::yz>({ 1500, 300, 600 });
    auto *A_yz2 = new float_array<3, reshape::yz>({ 1500, 300, 600 });
    A_yz1->distribute<3>({{partition::yz, 6}, {2, 1, NONE}});
    A_yz2->distribute<3>({{partition::yz, 6}, {1, 2, NONE}});
    delete A_yz1; delete A_yz2;

    return 0;
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
