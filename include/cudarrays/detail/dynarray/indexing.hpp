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
#ifndef CUDARRAYS_DETAIL_DYNARRAY_INDEXING_HPP_
#define CUDARRAYS_DETAIL_DYNARRAY_INDEXING_HPP_

#include "../../common.hpp"
#include "../../utils.hpp"

namespace cudarrays {

enum indexer_type {
    INDEX_NAIVE = 1,
    INDEX_VM    = 2,
    INDEX_PTR_TABLE = 3
};

template <unsigned Dims, unsigned Idx>
struct linearizer_impl {
    static constexpr unsigned FirstDim = 3 - Dims;
    static __host__ __device__ inline
    array_index_t access_pos(const array_size_t *offs,
                             const array_index_t &idx1, const array_index_t &idx2, const array_index_t &idx3)
    {
        array_index_t ret;
        if (Idx == 0)
            ret = offs[0 - FirstDim] * idx1;
        else if (Idx == 1)
            ret = offs[1 - FirstDim] * idx2;
        else // Not going to happen
            ret = -1;

        return ret + linearizer_impl<Dims, Idx + 1>::access_pos(offs, idx1, idx2, idx3);
    }
};

template <unsigned Dims>
struct linearizer_impl<Dims, 2> {
    static __host__ __device__ inline
    array_index_t access_pos(const array_size_t *,
                             const array_index_t &, const array_index_t &, const array_index_t &idx3)
    {
        return idx3;
    }
};

template <unsigned Dims>
struct linearizer {
    static constexpr unsigned FirstDim = 3 - Dims;

    static __host__ __device__ inline
    array_index_t access_pos(const array_size_t *offs,
                             const array_index_t &idx1, const array_index_t &idx2, const array_index_t &idx3)
    {
        return linearizer_impl<Dims, FirstDim>::access_pos(offs, idx1, idx2, idx3);
    }
};

template <typename PartConf>
struct index_block {
    static constexpr unsigned Dims     = PartConf::dimensions;
    static constexpr unsigned FirstDim = 3 - Dims;

    template <bool Part>
    static __host__ __device__ inline
    array_index_t local_idx(array_index_t idx, const array_size_t &elemsDim)
    {
        return Part? idx % elemsDim: idx;
    }

    template <bool Part>
    static __host__ __device__ inline
    array_index_t proc_off(array_index_t idx, const array_size_t &elemsDim, const array_size_t &elemsChunk)
    {
        return Part? (idx / elemsDim) * elemsChunk: 0;
    }

    static __host__ __device__ inline
    array_index_t access_pos(const array_size_t offs[Dims - 1],
                             const array_size_t elems[Dims],
                             const array_size_t offsProcs[Dims],
                             array_index_t idx1, array_index_t idx2, array_index_t idx3)
    {
        using my_linearizer = linearizer<Dims>;

        auto local = my_linearizer::access_pos(offs,
                                               local_idx<PartConf::Z>(idx1, elems[0 - FirstDim]),
                                               local_idx<PartConf::Y>(idx2, elems[1 - FirstDim]),
                                               local_idx<PartConf::X>(idx3, elems[2 - FirstDim]));
        auto global = proc_off<PartConf::Z>(idx1, elems[0 - FirstDim], offsProcs[0 - FirstDim]) +
                      proc_off<PartConf::Y>(idx2, elems[1 - FirstDim], offsProcs[1 - FirstDim]) +
                      proc_off<PartConf::X>(idx3, elems[2 - FirstDim], offsProcs[2 - FirstDim]);

        return local + global;
    }
};


template <typename PartConf>
struct index_cyclic {
    static constexpr unsigned Dims     = PartConf::dimensions;
    static constexpr unsigned FirstDim = 3 - Dims;

    template <bool Part>
    static __host__ __device__ inline
    array_index_t local_idx(array_index_t idx, const array_size_t &procsDim)
    {
        if (Part) return idx / procsDim;
        else      return idx;
    }

    template <bool Part>
    static __host__ __device__ inline
    array_index_t proc_off(array_index_t idx, const array_size_t &procsDim, const array_size_t &elemsChunk)
    {
        if (Part) return (idx % procsDim) * elemsChunk;
        else      return 0;
    }

    static __host__ __device__ inline
    array_index_t access_pos(const array_size_t offs[Dims - 1],
                             const array_size_t procs[Dims],
                             const array_size_t offsProcs[Dims],
                             array_index_t idx1, array_index_t idx2, array_index_t idx3)
    {
        using my_linearizer = linearizer<Dims>;

        array_index_t local;
        local = my_linearizer::access_pos(offs,
                                          local_idx<PartConf::Z>(idx1, procs[0 - FirstDim]),
                                          local_idx<PartConf::Y>(idx2, procs[1 - FirstDim]),
                                          local_idx<PartConf::X>(idx3, procs[2 - FirstDim]));
        array_index_t global;
        global = proc_off<PartConf::Z>(idx1, procs[0 - FirstDim], offsProcs[0 - FirstDim]) +
                 proc_off<PartConf::Y>(idx2, procs[1 - FirstDim], offsProcs[1 - FirstDim]) +
                 proc_off<PartConf::X>(idx3, procs[2 - FirstDim], offsProcs[2 - FirstDim]);

        return local + global;
    }
};

template <typename PartConf, unsigned BlockSize>
struct index_block_cyclic {
    static constexpr unsigned Dims     = PartConf::dimensions;
    static constexpr unsigned FirstDim = 3 - Dims;

    template <bool Part>
    static __host__ __device__ inline
    array_index_t local_idx(array_index_t idx, const array_size_t &blockDim)
    {
        if (Part) return idx % blockDim;
        else      return idx;
    }

    template <bool Part>
    static __host__ __device__ inline
    array_index_t proc_off(array_index_t idx, const array_size_t &blockDim, const array_size_t &procs, const array_size_t &elemsChunk)
    {
        if (Part) return ((idx / blockDim) % procs) * elemsChunk;
        else      return 0;
    }

    template <bool Part>
    static __host__ __device__ inline
    array_index_t block_idx(array_index_t idx, const array_size_t &blockDim, const array_size_t &procs)
    {
        if (Part) return idx/(blockDim * procs);
        else      return 0;
    }

    static __host__ __device__ inline
    array_index_t access_pos(const array_size_t offs[Dims - 1],
                             const array_size_t blocks[Dims],
                             const array_size_t blockDims[Dims],
                             const array_size_t procs[Dims],
                             const array_size_t offsProcs[Dims],
                             array_index_t idx1, array_index_t idx2, array_index_t idx3)
    {
        using block_linearizer = linearizer<Dims>;

        array_index_t local;
        local = block_linearizer::access_pos(offs,
                                             local_idx<PartConf::Z>(idx1, blockDims[0 - FirstDim]),
                                             local_idx<PartConf::Y>(idx2, blockDims[1 - FirstDim]),
                                             local_idx<PartConf::X>(idx3, blockDims[2 - FirstDim]));

        using my_linearizer = linearizer<Dims>;

        array_index_t block;
        block = my_linearizer::access_pos(offs,
                                          block_idx<PartConf::Z>(idx1, blockDims[0 - FirstDim], procs[0 - FirstDim]),
                                          block_idx<PartConf::Y>(idx2, blockDims[1 - FirstDim], procs[1 - FirstDim]),
                                          block_idx<PartConf::X>(idx3, blockDims[2 - FirstDim], procs[2 - FirstDim]));
        array_index_t global;
        global = proc_off<PartConf::Z>(idx1, blocks[0 - FirstDim], procs[0 - FirstDim], offsProcs[0 - FirstDim]) +
                 proc_off<PartConf::Y>(idx2, blocks[1 - FirstDim], procs[1 - FirstDim], offsProcs[1 - FirstDim]) +
                 proc_off<PartConf::X>(idx3, blocks[2 - FirstDim], procs[2 - FirstDim], offsProcs[2 - FirstDim]);

        return local + block + global;
    }
};

#if 0
template <typename T, unsigned Dims, bool PartZ, bool PartY, bool PartX, unsigned Idx>
struct index_ptr {
    static constexpr unsigned FirstDim = 3 - Dims;

    static __host__ __device__ inline
    array_index_t access_chunk(const array_size_t chunks_offs[Dims],
                               const array_size_t elems[Dims],
                               array_index_t idx1, array_index_t idx2, array_index_t idx3)
    {
        if (Idx == 0) {
            return (PartZ? (idx1 / elems[Idx - (3 - Dims)]) * chunks_offs[Idx] : 1) +
                                          index_ptr<T, Dims, PartZ, PartY, PartX, Idx + 1>::access_chunk(chunks_offs,
                                                                                                         elems,
                                                                                                         idx1, idx2, idx3);
        } else if (Idx == 1) {
            return (PartY? (idx2 / elems[Idx - (3 - Dims)]) * chunks_offs[Idx] : 1) +
                                          index_ptr<T, Dims, PartZ, PartY, PartX, Idx + 1>::access_chunk(chunks_offs,
                                                                                                         elems,
                                                                                                         idx1, idx2, idx3);
        } else if (Idx == 2) {
            return (PartX? (idx3 / elems[Idx - (3 - Dims)]) * chunks_offs[Idx] : 1) +
                                          index_ptr<T, Dims, PartZ, PartY, PartX, Idx + 1>::access_chunk(chunks_offs,
                                                                                                         elems,
                                                                                                         idx1, idx2, idx3);
        } else {
            return 0;
        }
    }

    template <bool Part>
    static __host__ __device__ inline
    array_index_t local_idx(array_index_t idx, array_index_t elemsDim)
    {
        if (Part) return idx % elemsDim;
        else      return idx;
    }

    static __host__ __device__ inline
    array_index_t access_local(const array_size_t offs[Dims],
                               const array_size_t elems[Dims],
                               array_index_t idx1, array_index_t idx2, array_index_t idx3)
    {

        return indexer<T, Dims, INDEX_VM, Idx>::access_pos(offs, local_idx<PartZ>(idx1, elems[0 - (3 - Dims)]),
                                                                 local_idx<PartY>(idx2, elems[1 - (3 - Dims)]),
                                                                 local_idx<PartX>(idx3, elems[2 - (3 - Dims)]));
    }
};

template <typename T, unsigned Dims, bool PartZ, bool PartY, bool PartX>
struct index_ptr<T, Dims, PartZ, PartY, PartX, 3>
{
    static __host__ __device__ inline
    array_index_t access_chunk(const array_size_t [Dims],
                               const array_size_t [Dims],
                               array_index_t , array_index_t , array_index_t )
    {
        return 0;
    }
};
#endif

}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
