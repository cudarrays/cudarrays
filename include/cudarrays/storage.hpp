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
#ifndef CUDARRAYS_STORAGE_HPP_
#define CUDARRAYS_STORAGE_HPP_

#include <sys/mman.h>
#include <unistd.h>

#include <array>
#include <functional>
#include <type_traits>

#include "coherence.hpp"
#include "common.hpp"
#include "compiler.hpp"
#include "utils.hpp"

namespace cudarrays {

namespace layout {
struct rmo {};
struct cmo {};

template <unsigned... Order>
struct custom {};
};

enum storage_impl {
    AUTO                 = 0,
    RESHAPE_BLOCK        = 2,
    RESHAPE_CYCLIC       = 3,
    RESHAPE_BLOCK_CYCLIC = 4,
    VM                   = 5,
    REPLICATED           = 10,
};

#if 0
template <bool... PartVals>
struct part_impl {
    static constexpr size_t Dims = sizeof...(PartVals);
    using array_type = array_dev<bool, sizeof...(PartVals)>;
    static constexpr array_type Pos{PartVals...};
};
#else
template <bool... PartVals>
struct part_impl;

template <bool PartX>
struct part_impl<PartX> {
    static constexpr unsigned dimensions = 1;
    static constexpr bool X = PartX;

    static constexpr bool Y = false;
    static constexpr bool Z = false;
};

template <bool PartY, bool PartX>
struct part_impl<PartY, PartX> {
    static constexpr unsigned dimensions = 2;
    static constexpr bool X = PartX;
    static constexpr bool Y = PartY;

    static constexpr bool Z = false;
};

template <bool PartZ, bool PartY, bool PartX>
struct part_impl<PartZ, PartY, PartX> {
    static constexpr unsigned dimensions = 3;
    static constexpr bool X = PartX;
    static constexpr bool Y = PartY;
    static constexpr bool Z = PartZ;
};

#endif

// Predefined partition classes
template <unsigned Dims>
struct part_none;
template <unsigned Dims>
struct part_x;
template <unsigned Dims>
struct part_y;
template <unsigned Dims>
struct part_z;
template <unsigned Dims>
struct part_xy;
template <unsigned Dims>
struct part_xz;
template <unsigned Dims>
struct part_yz;
template <unsigned Dims>
struct part_xyz;

template <>
struct part_none<1> {
    using type = part_impl<false>;
};
template <>
struct part_none<2> {
    using type = part_impl<false, false>;
};
template <>
struct part_none<3> {
    using type = part_impl<false, false, false>;
};

template <>
struct part_x<1> {
    using type = part_impl<true>;
};
template <>
struct part_x<2> {
    using type = part_impl<false, true>;
};
template <>
struct part_x<3> {
    using type = part_impl<false, false, true>;
};

template <>
struct part_y<2> {
    using type = part_impl<true, false>;
};
template <>
struct part_y<3> {
    using type = part_impl<false, true, false>;
};

template <>
struct part_z<3> {
    using type = part_impl<true, false, false>;
};

template <>
struct part_xy<2> {
    using type = part_impl<true, true>;
};
template <>
struct part_xy<3> {
    using type = part_impl<false, true, true>;
};

template <>
struct part_xz<3> {
    using type = part_impl<true, false, true>;
};

template <>
struct part_yz<3> {
    using type = part_impl<true, true, false>;
};

template <>
struct part_xyz<3> {
    using type = part_impl<true, true, true>;
};

template <unsigned... PosVals>
struct storage_reorder_conf;

// Predefined reorder classes
template <unsigned Dims>
struct storage_reorder_none;

template <>
struct storage_reorder_none<1u> {
    using type = storage_reorder_conf<0u>;
};
template <>
struct storage_reorder_none<2u> {
    using type = storage_reorder_conf<0u, 1u>;
};
template <>
struct storage_reorder_none<3u> {
    using type = storage_reorder_conf<0u, 1u, 2u>;
};

template <unsigned Dims>
struct storage_reorder_inverse;

template <>
struct storage_reorder_inverse<1u> {
    using type = storage_reorder_conf<0u>;
};
template <>
struct storage_reorder_inverse<2u> {
    using type = storage_reorder_conf<1u, 0u>;
};
template <>
struct storage_reorder_inverse<3u> {
    using type = storage_reorder_conf<2u, 1u, 0u>;
};

template <unsigned Dims, typename StorageType>
struct make_reorder;

template <unsigned Dims>
struct make_reorder<Dims, layout::rmo> {
    using type = typename storage_reorder_none<Dims>::type;
};

template <unsigned Dims>
struct make_reorder<Dims, layout::cmo> {
    using type = typename storage_reorder_inverse<Dims>::type;
};

template <unsigned Dims, unsigned... Order>
struct make_reorder<Dims, layout::custom<Order...>> {
    using type = storage_reorder_conf<Order...>;
};

template <storage_impl Storage>
struct select_auto_impl {
    static constexpr storage_impl impl = Storage;
};

template <>
struct select_auto_impl<AUTO> {
    static constexpr storage_impl impl = REPLICATED;
};

template <storage_impl Storage, template <unsigned> class PartConf>
struct storage_conf {
    static constexpr storage_impl impl = Storage;
    static constexpr storage_impl final_impl = select_auto_impl<Storage>::impl;
    template <unsigned Dims>
    using part_type = typename PartConf<Dims>::type;
};

template <storage_impl Impl>
struct storage_part
{
    static constexpr storage_impl impl = Impl;
    static constexpr storage_impl final_impl = select_auto_impl<Impl>::impl;

    using none = storage_conf<Impl, part_none>;

    using x = storage_conf<Impl, part_x>;
    using y = storage_conf<Impl, part_y>;
    using z = storage_conf<Impl, part_z>;

    using xy = storage_conf<Impl, part_xy>;
    using xz = storage_conf<Impl, part_xz>;
    using yz = storage_conf<Impl, part_yz>;

    using xyz = storage_conf<Impl, part_xyz>;
};

struct tag_auto : storage_part<AUTO> {
    static constexpr const char *name = "AUTO";
};

struct reshape : storage_part<RESHAPE_BLOCK> {
    static constexpr const char *name = "Reshape BLOCK";
};

using reshape_block = reshape;

struct reshape_cyclic : storage_part<RESHAPE_CYCLIC> {
    static constexpr const char *name = "Reshape CYCLIC";
};

struct reshape_block_cyclic : storage_part<RESHAPE_BLOCK_CYCLIC> {
    static constexpr const char *name = "Reshape BLOCK-CYCLIC";
};

struct vm : storage_part<VM> {
    static constexpr const char *name = "Virtual Memory";
};

struct replicate : storage_part<REPLICATED> {
    static constexpr const char *name = "Replicate";
};

enum class compute {
    none,
    x, y, z,
    xy, xz, yz,
    xyz
};

template <unsigned Dims>
struct compute_part_helper;

template <>
struct compute_part_helper<1> {
    static std::array<bool, 1>
    make_array(compute c)
    {
        std::array<bool, 1> ret;
        if (c == compute::none)   ret = { false };
        else if (c == compute::x) ret = { true  };
        else abort();
        return ret;
    }
};

template <>
struct compute_part_helper<2> {
    static std::array<bool, 2>
    make_array(compute c)
    {
        std::array<bool, 2> ret;
        if (c == compute::none)    ret = { false, false };
        else if (c == compute::x)  ret = { false, true  };
        else if (c == compute::y)  ret = { true, false  };
        else if (c == compute::xy) ret = { true, true   };
        else abort();
        return ret;
    }
};

template <>
struct compute_part_helper<3> {
    static std::array<bool, 3>
    make_array(compute c)
    {
        std::array<bool, 3> ret;
        if (c == compute::none)     ret = { false, false, false };
        else if (c == compute::x)   ret = { false, false, true  };
        else if (c == compute::y)   ret = { false, true,  false };
        else if (c == compute::z)   ret = { true,  false, false };
        else if (c == compute::xy)  ret = { false, true,  true  };
        else if (c == compute::xz)  ret = { true,  false, true  };
        else if (c == compute::yz)  ret = { true,  true,  false };
        else if (c == compute::xyz) ret = { true,  true,  true  };
        else abort();
        return ret;
    }
};

template <unsigned Dims>
struct compute_conf {
    std::array<bool, Dims> info;
    unsigned procs;

    compute_conf(const std::array<bool, Dims> &_info, unsigned _procs = 0) :
        info(_info),
        procs(_procs)
    {
    }

    compute_conf(compute c, unsigned _procs) :
        procs(_procs)
    {
        info = compute_part_helper<Dims>::make_array(c);
    }

    constexpr bool is_dim_part(unsigned dim) const
    {
        return info[dim];
    }

    unsigned get_part_dims() const
    {
        return utils::count(info, true);
    }
};

static constexpr int DimInvalid = -1;

template <unsigned DimsComp, unsigned Dims>
struct compute_mapping {
    compute_conf<DimsComp> comp;
    std::array<int, Dims> info;

    unsigned get_array_part_dims() const
    {
        return utils::count_if(info, [](int m) { return m != DimInvalid; });
    }

    bool is_array_dim_part(unsigned dim) const
    {
        return info[dim] != DimInvalid;
    }

    std::array<int, Dims> get_array_to_comp() const
    {
        std::array<int, Dims> ret;
        // Register the mapping
        for (auto i : utils::make_range(Dims)) {
            ret[i] = is_array_dim_part(i)? int(DimsComp) - (info[i] + 1):
                                           DimInvalid;
        }
        return ret;
    }
};

/**
 * Create a DimsComp-dimensional GPU grid the lowest order dimensions are maximized
 */
template <unsigned DimsComp>
static std::array<unsigned, DimsComp>
helper_distribution_get_gpu_grid(const compute_conf<DimsComp> &comp)
{
    std::array<unsigned, DimsComp> gpuGrid;

    // Check if we can map the arrayPartitionGrid on the GPUs
    std::vector<unsigned> factorsGpus = utils::get_factors(comp.procs);
    std::sort(factorsGpus.begin(), factorsGpus.end(), std::greater<unsigned>());

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
    std::array<unsigned, Dims> ret;

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
    std::array<array_size_t, Dims> ret;

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
    std::array<array_size_t, Dims - 1> ret;

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
    std::array<array_size_t, Dims> ret;

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
    std::array<unsigned, DimsComp> ret;

    unsigned gridOff = 1;
    for (int dim = DimsComp - 1; dim >= 0; --dim) {
        ret[dim] = gridOff;
        if (dim < DimsComp) {
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
    std::array<unsigned, Dims> ret;

    for (unsigned dim : utils::make_range(Dims)) {
        int compDim = arrayDimToCompDim[dim];

        ret[dim] = (compDim != DimInvalid)? gpuGridOffs[compDim]: 0;
    }

    return ret;
}

}

#endif
/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
