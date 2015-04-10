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

template <typename T, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence>
using dynarray1d = dynarray<T, 1, layout::rmo, PartConf, CoherencePolicy>;
template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence>
using dynarray2d = dynarray<T, 2, StorageType, PartConf, CoherencePolicy>;
template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence>
using dynarray3d = dynarray<T, 3, StorageType, PartConf, CoherencePolicy>;

template <typename T, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence>
using vector = dynarray1d<T, PartConf, CoherencePolicy>;

template <typename T, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence>
using vector_ref = dynarray_ref<vector<T, PartConf, CoherencePolicy>>;
template <typename T, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence>
using vector_cref = dynarray_cref<vector<T, PartConf, CoherencePolicy>>;

#define TYPE_VECTOR(n,t) \
template <typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence> \
using vector_##n = vector<t, PartConf, CoherencePolicy>;                                                     \
template <typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence> \
using vector_##n##_ref = vector_ref<vector<t, PartConf, CoherencePolicy>>;                                   \
template <typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence> \
using vector_##n##_cref = vector_cref<vector<t, PartConf, CoherencePolicy>>;

TYPE_VECTOR(b,bool)
TYPE_VECTOR(i,int)
TYPE_VECTOR(u,unsigned)
TYPE_VECTOR(f32,float)
TYPE_VECTOR(f64,double)

template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence>
using matrix       = dynarray2d<T, StorageType, PartConf, CoherencePolicy>;

template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence>
using matrix_ref   = dynarray_ref<matrix<T, StorageType, PartConf, CoherencePolicy>>;
template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence>
using matrix_cref  = dynarray_cref<matrix<T, StorageType, PartConf, CoherencePolicy>>;

#define TYPE_MATRIX(n,t) \
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence> \
using matrix_##n  = matrix<t, StorageType, PartConf, CoherencePolicy>;                                                                           \
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence> \
using matrix_##n##_ref  = matrix_ref<matrix<t, StorageType, PartConf, CoherencePolicy>>;                                                                           \
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence> \
using matrix_##n##_cref = matrix_ref<matrix<t, StorageType, PartConf, CoherencePolicy>>;

TYPE_MATRIX(b,bool)
TYPE_MATRIX(i,int)
TYPE_MATRIX(u,unsigned)
TYPE_MATRIX(f32,float)
TYPE_MATRIX(f64,double)

template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence>
using volume       = dynarray3d<T, StorageType, PartConf, CoherencePolicy>;

template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence>
using volume_ref   = dynarray_ref<volume<T, StorageType, PartConf, CoherencePolicy>>;
template <typename T, typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence>
using volume_cref  = dynarray_cref<volume<T, StorageType, PartConf, CoherencePolicy>>;

#define TYPE_VOLUME(n,t) \
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence> \
using volume_##n  = volume<t, StorageType, PartConf, CoherencePolicy>;                                                                           \
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence> \
using volume_##n##_ref  = volume_ref<volume<t, StorageType, PartConf, CoherencePolicy>>;                                                                           \
template <typename StorageType = layout::rmo, typename PartConf = tag_auto::none, template <typename> class CoherencePolicy = default_coherence> \
using volume_##n##_cref = volume_ref<volume<t, StorageType, PartConf, CoherencePolicy>>;

TYPE_VOLUME(b,bool)
TYPE_VOLUME(i,int)
TYPE_VOLUME(u,unsigned)
TYPE_VOLUME(f32,float)
TYPE_VOLUME(f64,double)


}

#endif
