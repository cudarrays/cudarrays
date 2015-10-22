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
#ifndef CUDARRAYS_DETAIL_DYNARRAY_HELPERS_HPP_
#define CUDARRAYS_DETAIL_DYNARRAY_HELPERS_HPP_

#include <array>
#include <vector>

#include "../../storage.hpp"
#include "../../compute.hpp"

namespace cudarrays {

/**
 * Create a DimsComp-dimensional GPU grid the lowest order dimensions are maximized
 */
template <unsigned DimsComp>
static std::array<unsigned, DimsComp>
helper_distribution_get_gpu_grid(const cudarrays::compute_conf<DimsComp> &comp)
{
    std::array<unsigned, DimsComp> gpuGrid = {{}};

    // Check if we can map the arrayPartitionGrid on the GPUs
    std::vector<unsigned> factorsGpus = utils::get_factors(comp.procs);
    utils::sort(factorsGpus, std::greater<unsigned>());

#if 0
    if (factorsGPUs.size() < compPartDims)
        FATAL("CUDArrays cannot partition %u dimensions into %u GPUs", arrayPartDims, comp.procs);
#endif

    // Create the GPU grid
    unsigned j = 0;
    for (unsigned i : utils::make_range(DimsComp)) {
        if (comp.procs > 1) {
            unsigned partition = 1;
            if (comp.is_dim_part(i)) {
                ASSERT(j < factorsGpus.size());

                std::vector<unsigned>::iterator pos = factorsGpus.begin() + j;
                size_t inc = (j == 0)? factorsGpus.size() - comp.get_part_dims() + 1: 1;

                // DEBUG("Reshape> ALLOC: Collapsing%u: %zd:%zd", i, j, j + inc);
                partition = std::accumulate(pos, pos + inc, 1, std::multiplies<unsigned>());
                j += inc;
            }
            gpuGrid[i] = partition;
        } else {
            gpuGrid[i] = 1;
        }
    }

    return gpuGrid;
}


template <size_t DimsComp, size_t Dims>
static std::array<unsigned, Dims>
helper_distribution_get_array_grid(const std::array<unsigned, DimsComp> &gpuGrid,
                                   const std::array<int, Dims>          &arrayDimToCompDim)
{
    std::array<unsigned, Dims> ret{{}};

    // Compute the array grid and the local sizes
    for (unsigned i : utils::make_range(Dims)) {
        int compDim = arrayDimToCompDim[i];

        unsigned partition = 1;
        if (compDim != DimInvalid) {
            partition = gpuGrid[compDim];
        } else {
            // TODO: REPLICATION
        }

        ret[i] = partition;
    }

    return ret;
}

template <size_t Dims>
static std::array<array_size_t, Dims>
helper_distribution_get_local_dims(const std::array<array_size_t, Dims> &dims,
                                   const std::array<unsigned, Dims>     &arrayGrid)
{
    std::array<array_size_t, Dims> ret{{}};

    // Compute the array grid and the local sizes
    for (unsigned i : utils::make_range(Dims)) {
        // TODO: REPLICATION
        ret[i] = utils::div_ceil(dims[i], arrayGrid[i]);
    }

    return ret;
}

template <size_t Dims>
static array_size_t
helper_distribution_get_local_elems(const std::array<array_size_t, Dims> &dims,
                                    array_size_t boundary = 1)
{
    array_size_t ret;

    ret = utils::accumulate(dims, 1, std::multiplies<array_size_t>());
    // ... adjusting the tile size to VM SIZE
    ret = utils::round_next(ret, boundary);

    return ret;
}

template <size_t Dims>
static std::array<array_size_t, Dims - 1>
helper_distribution_get_local_offs(const std::array<array_size_t, Dims> &dims)
{
    std::array<array_size_t, Dims - 1> ret{{}};

    array_size_t off = 1;
    for (ssize_t dim = ssize_t(Dims) - 2; dim >= 0; --dim) {
        off *= dims[dim + 1];
        ret[dim] = off;
    }

    return ret;
}

template <size_t Dims>
static std::array<array_size_t, Dims>
helper_distribution_get_intergpu_offs(array_size_t elemsLocal,
                                      const std::array<unsigned, Dims> &arrayGrid,
                                      const std::array<int, Dims>      &arrayDimToCompDim)
{
    std::array<array_size_t, Dims> ret{{}};

    array_size_t off = 1;
    for (ssize_t dim = Dims - 1; dim >= 0; --dim) {
        if (arrayDimToCompDim[dim] != DimInvalid) {
            ret[dim] = off * elemsLocal;
            off *= arrayGrid[dim];
        } else {
            ret[dim] = 0;
        }
    }

    return ret;
}

template <size_t DimsComp>
static std::array<unsigned, DimsComp>
helper_distribution_gpu_get_offs(const std::array<unsigned, DimsComp> &gpuGrid)
{
    std::array<unsigned, DimsComp> ret{{}};

    unsigned gridOff = 1;
    for (ssize_t dim = DimsComp - 1; dim >= 0; --dim) {
        ret[dim] = gridOff;
        if (dim < ssize_t(DimsComp)) {
            gridOff *= gpuGrid[dim];
        }
    }

    return ret;
}

template <size_t DimsComp, size_t Dims>
static std::array<unsigned, Dims>
helper_distribution_get_array_dim_to_gpus(const std::array<unsigned, DimsComp> &gpuGridOffs,
                                          const std::array<int, Dims>          &arrayDimToCompDim)
{
    std::array<unsigned, Dims> ret{{}};

    for (unsigned dim : utils::make_range(Dims)) {
        int compDim = arrayDimToCompDim[dim];

        ret[dim] = (compDim != DimInvalid)? gpuGridOffs[compDim]: 0;
    }

    return ret;
}

}

#endif
