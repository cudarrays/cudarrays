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
#ifndef CUDARRAYS_TYPES_HPP_
#define CUDARRAYS_TYPES_HPP_

#include "dynarray.hpp"

namespace cudarrays {

template <typename T, typename PartConf = tag_auto::none>
using dynarray1d       = dynarray<T, 1, false, layout::rmo, PartConf>;
template <typename T, typename PartConf = tag_auto::none>
using const_dynarray1d = dynarray<T, 1, true, layout::rmo, PartConf>;
template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using dynarray2d       = dynarray<T, 2, false, StorageType, PartConf>;
template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_dynarray2d = dynarray<T, 2, true, StorageType, PartConf>;
template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using dynarray3d       = dynarray<T, 3, false, StorageType, PartConf>;
template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_dynarray3d = dynarray<T, 3, true, StorageType, PartConf>;

template <typename T, typename PartConf = tag_auto::none>
using vector       = dynarray1d<T, PartConf>;
template <typename T, typename PartConf = tag_auto::none>
using const_vector = const_dynarray1d<T, PartConf>;

template <typename PartConf = tag_auto::none>
using vector_b = vector<bool, PartConf>;
template <typename PartConf = tag_auto::none>
using vector_i = vector<int, PartConf>;
template <typename PartConf = tag_auto::none>
using vector_u = vector<unsigned, PartConf>;
template <typename PartConf = tag_auto::none>
using vector_f32 = vector<float, PartConf>;
template <typename PartConf = tag_auto::none>
using vector_f64 = vector<double, PartConf>;

template <typename PartConf = tag_auto::none>
using const_vector_b = const_vector<bool, PartConf>;
template <typename PartConf = tag_auto::none>
using const_vector_i = const_vector<int, PartConf>;
template <typename PartConf = tag_auto::none>
using const_vector_u = const_vector<unsigned, PartConf>;
template <typename PartConf = tag_auto::none>
using const_vector_f32 = const_vector<float, PartConf>;
template <typename PartConf = tag_auto::none>
using const_vector_f64= const_vector<double, PartConf>;

template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using matrix       = dynarray2d<T, StorageType, PartConf>;
template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_matrix = const_dynarray2d<T, StorageType, PartConf>;

template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using matrix_b     = matrix<bool, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using matrix_i     = matrix<int, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using matrix_u     = matrix<unsigned, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using matrix_f32   = matrix<float, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using matrix_f64   = matrix<double, PartConf>;

template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_matrix_b = const_matrix<bool, StorageType, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_matrix_i = const_matrix<int, StorageType, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_matrix_u = const_matrix<unsigned, StorageType, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_matrix_f32 = const_matrix<float, StorageType, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_matrix_f64= const_matrix<double, StorageType, PartConf>;

template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using volume       = dynarray3d<T, StorageType, PartConf>;
template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_volume = const_dynarray3d<T, StorageType, PartConf>;

template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using volume_b     = volume<bool, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using volume_i     = volume<int, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using volume_u     = volume<unsigned, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using volume_f32   = volume<float, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using volume_f64   = volume<double, PartConf>;

template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_volume_b = const_volume<bool, StorageType, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_volume_i = const_volume<int, StorageType, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_volume_u = const_volume<unsigned, StorageType, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_volume_f32 = const_volume<float, StorageType, PartConf>;
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none>
using const_volume_f64 = const_volume<double, StorageType, PartConf>;

}

#endif
