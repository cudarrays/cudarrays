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

template <typename OffsetsSeq, unsigned Dim = 0>
struct linearizer_hybrid {
    template <typename... Idxs>
    static __host__ __device__ inline
    array_index_t access_pos(const array_size_t *offs, const array_index_t &idx, const Idxs &...idxs)
    {
        array_index_t ret;
        constexpr auto Offset = SEQ_AT(OffsetsSeq, Dim);
        if (std::is_same<std::integral_constant<array_size_t, Offset>,
                         std::integral_constant<array_size_t, 0>>::value) {
            ret = offs[Dim] * idx;
        } else {
            ret = Offset * idx;
        }

        return ret + linearizer_hybrid<OffsetsSeq, Dim + 1>::access_pos(offs, idxs...);
    }

    static __host__ __device__ inline
    array_index_t access_pos(const array_size_t *, const array_index_t &idx)
    {
        return idx;
    }
};

struct indexer_utils {

template <typename... Idxs>
__host__ __device__ inline
static constexpr
array_index_t sum(array_index_t idx, Idxs... idxs)
{
    return idx + sum(idxs...);
}
 __host__ __device__ inline
static constexpr
array_index_t sum(array_index_t idx)
{
    return idx;
}

};

template <typename OffsetsSeq, typename PartSeq, typename DimIdxSeq>
struct index_block_detail;

template <typename OffsetsSeq, bool... PartSeq, unsigned... DimIdxSeq>
struct index_block_detail<OffsetsSeq,
                          SEQ_WITH_TYPE(bool, PartSeq...),
                          SEQ_WITH_TYPE(unsigned, DimIdxSeq...)> {
    static constexpr unsigned Dims = sizeof...(PartSeq);

    template <bool Part>
    __host__ __device__ inline
    static constexpr
    array_index_t local_idx(array_index_t idx, array_size_t elemsDim)
    {
        return Part? idx % elemsDim: idx;
    }

    template <bool Part>
    __host__ __device__ inline
    static constexpr
    array_index_t proc_off(array_index_t idx, array_size_t elemsDim, array_size_t elemsChunk)
    {
        return Part? (idx / elemsDim) * elemsChunk: 0;
    }

    template <typename... Idxs>
    static __host__ __device__ inline
    array_index_t access_pos(const array_size_t offs[Dims - 1],
                             const array_size_t elems[Dims],
                             const array_size_t offsProcs[Dims],
                             Idxs... idxs)
    {
        using my_linearizer = linearizer_hybrid<OffsetsSeq>;

        auto local  = my_linearizer::access_pos(offs, local_idx<PartSeq>(idxs, elems[DimIdxSeq])...);
        auto global = indexer_utils::sum(proc_off<PartSeq>(idxs, elems[DimIdxSeq], offsProcs[DimIdxSeq])...);

        return local + global;
    }
};

template <typename OffsetsSeq, typename PartSeq>
using index_block = index_block_detail<OffsetsSeq, PartSeq, SEQ_GEN_INC(unsigned(SEQ_SIZE(PartSeq)))>;

template <typename OffsetsSeq, typename PartSeq, typename DimIdxSeq>
struct index_cyclic_detail;

template <typename OffsetsSeq, bool... PartSeq, unsigned... DimIdxSeq>
struct index_cyclic_detail<OffsetsSeq,
                           SEQ_WITH_TYPE(bool, PartSeq...),
                           SEQ_WITH_TYPE(unsigned, DimIdxSeq...)> {
    static constexpr unsigned Dims = sizeof...(PartSeq);

    template <bool Part>
    __host__ __device__ inline
    static constexpr
    array_index_t local_idx(array_index_t idx, array_size_t procsDim)
    {
        return Part? idx / procsDim:
                     idx;
    }

    template <bool Part>
    static __host__ __device__ inline
    array_index_t proc_off(array_index_t idx, array_size_t procsDim, array_size_t elemsChunk)
    {
        return Part? (idx % procsDim) * elemsChunk:
                     0;
    }

    template <typename... Idxs>
    static __host__ __device__ inline
    array_index_t access_pos(const array_size_t offs[Dims - 1],
                             const array_size_t procs[Dims],
                             const array_size_t offsProcs[Dims],
                             Idxs... idxs)
    {
        using my_linearizer = linearizer_hybrid<OffsetsSeq>;

        auto local  = my_linearizer::access_pos(offs, local_idx<PartSeq>(idxs, procs[DimIdxSeq])...);
        auto global = indexer_utils::sum(proc_off<PartSeq>(idxs, procs[DimIdxSeq], offsProcs[DimIdxSeq])...);

        return local + global;
    }
};

template <typename OffsetsSeq, typename PartSeq>
using index_cyclic = index_cyclic_detail<OffsetsSeq, PartSeq, SEQ_GEN_INC(unsigned(SEQ_SIZE(PartSeq)))>;

template <typename OffsetsSeq, typename PartSeq, typename DimIdxSeq, unsigned BlockSize>
struct index_block_cyclic_detail;

template <typename OffsetsSeq, bool... PartSeq, unsigned... DimIdxSeq, unsigned BlockSize>
struct index_block_cyclic_detail<OffsetsSeq,
                                 SEQ_WITH_TYPE(bool, PartSeq...),
                                 SEQ_WITH_TYPE(unsigned, DimIdxSeq...),
                                 BlockSize> {
    static constexpr unsigned Dims = sizeof...(PartSeq);

    template <bool Part>
    __host__ __device__ inline
    static constexpr
    array_index_t local_idx(array_index_t idx, array_size_t blockDim)
    {
        return Part? idx % blockDim:
                     idx;
    }

    template <bool Part>
    __host__ __device__ inline
    static constexpr
    array_index_t proc_off(array_index_t idx, array_size_t blockDim, array_size_t procs, array_size_t elemsChunk)
    {
        return Part? ((idx / blockDim) % procs) * elemsChunk:
                     0;
    }

    template <bool Part>
    __host__ __device__ inline
    static constexpr
    array_index_t block_idx(array_index_t idx, array_size_t blockDim, array_size_t procs)
    {
        return Part? idx/(blockDim * procs):
                     0;
    }

    template <typename... Idxs>
    static __host__ __device__ inline
    array_index_t access_pos(const array_size_t offs[Dims - 1],
                             const array_size_t blocks[Dims],
                             const array_size_t blockDims[Dims],
                             const array_size_t procs[Dims],
                             const array_size_t offsProcs[Dims],
                             Idxs... idxs)
    {
        using my_linearizer = linearizer_hybrid<OffsetsSeq>;

        auto local = my_linearizer::access_pos(offs, local_idx<PartSeq>(idxs, blockDims[DimIdxSeq])...);
        auto block = my_linearizer::access_pos(offs, block_idx<PartSeq>(idxs, blockDims[DimIdxSeq], procs[DimIdxSeq])...);
        auto global = indexer_utils::sum(proc_off<PartSeq>(idxs, blocks[DimIdxSeq], procs[DimIdxSeq], offsProcs[DimIdxSeq])...);

        return local + block + global;
    }
};

template <typename OffsetsSeq, typename PartSeq, unsigned BlockSize>
using index_block_cyclic = index_block_cyclic_detail<OffsetsSeq, PartSeq, SEQ_GEN_INC(unsigned(SEQ_SIZE(PartSeq))), BlockSize>;

}

#endif

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
