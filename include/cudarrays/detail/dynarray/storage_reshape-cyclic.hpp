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
#ifndef CUDARRAYS_DETAIL_DYNARRAY_STORAGE_RESHAPE_CYCLIC_HPP_
#define CUDARRAYS_DETAIL_DYNARRAY_STORAGE_RESHAPE_CYCLIC_HPP_

#include "../../utils.hpp"

#include "base.hpp"
#include "helpers.hpp"

namespace cudarrays {

template <typename StorageTraits>
class dynarray_storage<detail::storage_tag::RESHAPE_CYCLIC, StorageTraits> :
    public dynarray_base<StorageTraits>
{
    using base_storage_type = dynarray_base<StorageTraits>;
    using        value_type = typename base_storage_type::value_type;
    using    alignment_type = typename base_storage_type::alignment_type;
    using  dim_manager_type = typename base_storage_type::dim_manager_type;
    using host_storage_type = typename base_storage_type::host_storage_type;

    static constexpr auto dimensions = base_storage_type::dimensions;

    using PartConf = storage_part_dim_helper<StorageTraits::partition_value, dimensions>;

    using indexer_type = index_cyclic<typename StorageTraits::offsets_seq,
                                      typename StorageTraits::partitioning_seq>;

    __host__
    void alloc()
    {
        // Array grid layout
        unsigned partZ = (dimensions > 2)? arrayPartitionGrid_[dim_manager_type::DimIdxZ]: 1;
        unsigned partY = (dimensions > 1)? arrayPartitionGrid_[dim_manager_type::DimIdxY]: 1;
        unsigned partX =                   arrayPartitionGrid_[dim_manager_type::DimIdxX];
        // Array-to-GPU translation
        unsigned gpuDimForArrayZ = (dimensions > 2)? hostInfo_->arrayDimToGpus[dim_manager_type::DimIdxZ]: 1;
        unsigned gpuDimForArrayY = (dimensions > 1)? hostInfo_->arrayDimToGpus[dim_manager_type::DimIdxY]: 1;
        unsigned gpuDimForArrayX =                   hostInfo_->arrayDimToGpus[dim_manager_type::DimIdxX];

        DEBUG("ALLOCATE");
        // Iterate on all array grid dimensions
        for (unsigned pZ : utils::make_range(partZ)) {
            for (unsigned pY : utils::make_range(partY)) {
                for (unsigned pX : utils::make_range(partX)) {
                    // Compute the linear index of the array partition to be allocated
                    unsigned linear = pZ * partX * partY   + pY * partX           + pX;
                    // Compute the index of the GPU where the partition must be allocated
                    unsigned idx    = pZ * gpuDimForArrayZ + pY * gpuDimForArrayY + pX * gpuDimForArrayX;

                    DEBUG("in: %u,%u,%u -> %u", pZ, pY, pX, idx);

                    unsigned gpu = idx;
                    // Set the device where data is allocated
                    CUDA_CALL(cudaSetDevice(gpu));
                    // Perform memory allocation
                    value_type *tmp;
                    CUDA_CALL(cudaMalloc((void **) &tmp, hostInfo_->elemsLocal * sizeof(value_type)));
                    if (idx == 0) {
                        // Initialize the base address of the allocation
                        dataDev_ = tmp;
                    } else {
                        // Check that allocations are contiguous in the virtual address space
                        ASSERT(dataDev_ + linear * hostInfo_->elemsLocal == tmp);
                    }

                    DEBUG("- allocated %p (%zd) in GPU %u", tmp, hostInfo_->elemsLocal * sizeof(value_type), gpu);
                }
            }
        }

        dataDev_ += this->get_dim_manager().offset();
    }

public:
    template <unsigned DimsComp>
    __host__ void
    compute_distribution_internal_single(const cudarrays::compute_mapping<DimsComp, dimensions> &mapping)
    {
        std::array<int, dimensions> arrayDimToCompDim;
        std::array<unsigned, DimsComp> gpuGrid;

        // Register the mapping
        arrayDimToCompDim = mapping.get_array_to_comp();

        // Count partitioned dimensions in array and computation
        if (mapping.get_array_part_dims() > mapping.comp.get_part_dims())
            FATAL("Not enough partitioned comp dims: %u, to partition %u array dims",
                  mapping.comp.get_part_dims(),
                  mapping.get_array_part_dims());

#if 0
        // Check the minumum number of arrayPartitionGrid
        if ((1 << compPartDims) > mapping.comp.procs)
            FATAL("Not enough GPUs (%u), to partition %u", mapping.comp.procs, arrayPartDims);
#endif
        std::array<array_size_t, dimensions> dims = this->get_dim_manager().dims();

        // Distribute the partition uniformly across GPUs
        // 1- Compute GPU grid
        utils::fill(gpuGrid, 1);
        // 2- Compute array partitioning grid
        utils::fill(arrayPartitionGrid_, 1);
        // 3- Compute dimensions of each tile
        std::array<array_size_t, dimensions> localDims = helper_distribution_get_local_dims(dims, utils::make_array(arrayPartitionGrid_));
        utils::copy(localDims, hostInfo_->localDims_);
        // 4- Compute local offsets for the indexing functions
        std::array<array_size_t, dimensions - 1> localOffs = helper_distribution_get_local_offs(localDims);
        utils::copy(localOffs, localOffs_);
        // 5- Compute elements of each tile
        array_size_t elemsLocal = helper_distribution_get_local_elems(localDims, system::vm_cuda_align_elems<value_type>());
        hostInfo_->elemsLocal = elemsLocal;
        // 6- Compute the inter-GPU array offsets for each dimension (iterate from lowest-order dimension)
        std::array<array_size_t, dimensions> gpuOffs = helper_distribution_get_intergpu_offs(elemsLocal, utils::make_array(arrayPartitionGrid_), arrayDimToCompDim);
        utils::copy(gpuOffs, gpuOffs_);
        // 7- Compute the GPU grid offsets (iterate from lowest-order dimension)
        std::array<unsigned, DimsComp> gpuGridOffs = helper_distribution_gpu_get_offs(gpuGrid);
        // 8- Compute the array to GPU mapping needed for the allocation based on the grid offsets
        std::array<unsigned, dimensions> arrayDimToGpus = helper_distribution_get_array_dim_to_gpus(gpuGridOffs, arrayDimToCompDim);
        utils::copy(arrayDimToGpus, hostInfo_->arrayDimToGpus);

        DEBUG("BASE INFO");
        DEBUG("- array dims: %s", dims);
        DEBUG("- comp  dims: %u", DimsComp);
        DEBUG("- comp -> array: %s", arrayDimToCompDim);

        DEBUG("PARTITIONING");
        DEBUG("- gpus: %u", mapping.comp.procs);
        DEBUG("- comp  part: %s (%u)", mapping.comp.info, mapping.comp.get_part_dims());
        DEBUG("- comp  grid: %s", gpuGrid);
        DEBUG("- array grid: %s", arrayPartitionGrid_);
        DEBUG("- local elems: %s (%zd)", hostInfo_->localDims_, size_t(hostInfo_->elemsLocal));
        DEBUG("- local offs: %s", localOffs_);

        DEBUG("- array grid offsets: %s", hostInfo_->arrayDimToGpus);
        DEBUG("- gpu   grid offsets: %s", gpuOffs_);
    }

    template <unsigned DimsComp>
    __host__ void
    compute_distribution_internal(const cudarrays::compute_mapping<DimsComp, dimensions> &mapping)
    {
        std::array<int, dimensions> arrayDimToCompDim;
        std::array<unsigned, DimsComp> gpuGrid;

        // Register the mapping
        arrayDimToCompDim = mapping.get_array_to_comp();

        // Count partitioned dimensions in array and computation
        if (mapping.get_array_part_dims() > mapping.comp.get_part_dims())
            FATAL("Not enough partitioned comp dims: %u, to partition %u array dims",
                  mapping.comp.get_part_dims(),
                  mapping.get_array_part_dims());

#if 0
        // Check the minumum number of arrayPartitionGrid
        if ((1 << compPartDims) > mapping.comp.procs)
            FATAL("Not enough GPUs (%u), to partition %u", mapping.comp.procs, arrayPartDims);
#endif
        std::array<array_size_t, dimensions> dims = this->get_dim_manager().dims();

        // Distribute the partition uniformly across GPUs
        // 1- Compute GPU grid
        gpuGrid = helper_distribution_get_gpu_grid(mapping.comp);
        // 2- Compute array partitioning grid
        std::array<unsigned, dimensions> arrayPartitionGrid = helper_distribution_get_array_grid(gpuGrid, arrayDimToCompDim);
        utils::copy(arrayPartitionGrid, arrayPartitionGrid_);
        // 3- Compute dimensions of each tile
        std::array<array_size_t, dimensions> localDims = helper_distribution_get_local_dims(dims, arrayPartitionGrid);
        utils::copy(localDims, hostInfo_->localDims_);
        // 4- Compute local offsets for the indexing functions
        std::array<array_size_t, dimensions - 1> localOffs = helper_distribution_get_local_offs(localDims);
        utils::copy(localOffs, localOffs_);
        // 5- Compute elements of each tile
        array_size_t elemsLocal = helper_distribution_get_local_elems(localDims, system::vm_cuda_align_elems<value_type>());
        hostInfo_->elemsLocal = elemsLocal;
        // 6- Compute the inter-GPU array offsets for each dimension (iterate from lowest-order dimension)
        std::array<array_size_t, dimensions> gpuOffs = helper_distribution_get_intergpu_offs(elemsLocal, arrayPartitionGrid, arrayDimToCompDim);
        utils::copy(gpuOffs, gpuOffs_);
        // 7- Compute the GPU grid offsets (iterate from lowest-order dimension)
        std::array<unsigned, DimsComp> gpuGridOffs = helper_distribution_gpu_get_offs(gpuGrid);
        // 8- Compute the array to GPU mapping needed for the allocation based on the grid offsets
        std::array<unsigned, dimensions> arrayDimToGpus = helper_distribution_get_array_dim_to_gpus(gpuGridOffs, arrayDimToCompDim);
        utils::copy(arrayDimToGpus, hostInfo_->arrayDimToGpus);

        DEBUG("BASE INFO");
        DEBUG("- array dims: %s", dims);
        DEBUG("- comp  dims: %u", DimsComp);
        DEBUG("- comp -> array: %s", arrayDimToCompDim);

        DEBUG("PARTITIONING");
        DEBUG("- gpus: %u", mapping.comp.procs);
        DEBUG("- comp  part: %s (%u)", mapping.comp.info, mapping.comp.get_part_dims());
        DEBUG("- comp  grid: %s", gpuGrid);
        DEBUG("- array grid: %s", arrayPartitionGrid_);
        DEBUG("- local elems: %s (%zd)", hostInfo_->localDims_, size_t(hostInfo_->elemsLocal));
        DEBUG("- local offs: %s", localOffs_);

        DEBUG("- array grid offsets: %s", hostInfo_->arrayDimToGpus);
        DEBUG("- gpu   grid offsets: %s", gpuOffs_);
    }

    template <unsigned DimsComp>
    __host__ void
    compute_distribution(const cudarrays::compute_mapping<DimsComp, dimensions> &mapping)
    {
        TRACE_FUNCTION();

        hostInfo_.reset(new storage_host_info(mapping.comp.procs));

        compute_distribution_internal(mapping);
    }

    template <unsigned DimsComp>
    __host__ bool
    distribute(const cudarrays::compute_mapping<DimsComp, dimensions> &mapping)
    {
        bool ret = false;

        // Only distribute the first time. Otherwise use redistribute
        if (!dataDev_) {
            TRACE_FUNCTION();

            hostInfo_.reset(new storage_host_info(mapping.comp.procs));

            if (mapping.comp.procs == 1)
                compute_distribution_internal_single(mapping);
            else {
                FATAL("Not implemented yet");
                compute_distribution_internal(mapping);
            }
            alloc();

            ret = true;
        }

        return ret;
    }

    __host__ bool
    distribute(const std::vector<unsigned> &)
    {
        return false;
    }

    __host__ bool
    is_distributed() const
    {
        return dataDev_ != nullptr;
    }

private:
    value_type *dataDev_;

    array_size_t arrayPartitionGrid_[dimensions];
    array_size_t localOffs_[dimensions - 1];
    array_size_t gpuOffs_[dimensions];

    struct storage_host_info {
        unsigned gpus;

        array_size_t elemsLocal;
        std::array<unsigned, dimensions> arrayDimToGpus;

        std::array<array_size_t, dimensions> localDims_;

        storage_host_info(unsigned _gpus) :
            gpus{_gpus},
            elemsLocal{0}
        {
        }
    };

    std::unique_ptr<storage_host_info> hostInfo_;

public:
    __host__
    dynarray_storage(const extents<dimensions> &ext) :
        base_storage_type(ext),
        dataDev_{nullptr}
    {
    }

    __host__
    virtual ~dynarray_storage()
    {
        if (dataDev_ != nullptr) {
            // Free device memory (1 chunk per GPU)
            for (unsigned idx = 0; idx < hostInfo_->gpus; ++idx) {
                DEBUG("- freeing %p", dataDev_ - this->get_dim_manager().offset() + hostInfo_->elemsLocal * idx);
                CUDA_CALL(cudaFree(dataDev_ - this->get_dim_manager().offset() + hostInfo_->elemsLocal * idx));
            }
        }
    }

    __host__
    void to_host(host_storage_type &host)
    {
        TRACE_FUNCTION();

        value_type *unaligned = host.addr();
        auto &dimMgr = this->get_dim_manager();

        unsigned partZ = (dimensions > 2)? arrayPartitionGrid_[dim_manager_type::DimIdxZ]: 1;
        unsigned partY = (dimensions > 1)? arrayPartitionGrid_[dim_manager_type::DimIdxY]: 1;
        unsigned partX =                   arrayPartitionGrid_[dim_manager_type::DimIdxX];

        cudaMemcpy3DParms myParms;
        memset(&myParms, 0, sizeof(myParms));
        myParms.dstPtr = make_cudaPitchedPtr(unaligned,
                                             sizeof(value_type) * dimMgr.dim_align(dim_manager_type::DimIdxX),
                                                                  dimMgr.dim_align(dim_manager_type::DimIdxX),
                                             dimensions > 1? dimMgr.dim_align(dim_manager_type::DimIdxY): 1);

        for (unsigned pZ : utils::make_range(partZ)) {
            for (unsigned pY : utils::make_range(partY)) {
                for (unsigned pX : utils::make_range(partX)) {
                    array_index_t localZ = dimensions > 2? hostInfo_->localDims_[dim_manager_type::DimIdxZ]: 1;
                    array_index_t localY = dimensions > 1? hostInfo_->localDims_[dim_manager_type::DimIdxY]: 1;
                    array_index_t localX =                 hostInfo_->localDims_[dim_manager_type::DimIdxX];

                    array_index_t blockOff = pZ * (dimensions > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0) +
                                             pY * (dimensions > 1? gpuOffs_[dim_manager_type::DimIdxY]: 0) +
                                             pX *                  gpuOffs_[dim_manager_type::DimIdxX];

                    DEBUG("TO_HOST: Src Block Off: %u", blockOff);

                    DEBUG("TO_HOST: Block (%u, %u, %u)", pZ, pY, pX);
                    DEBUG("TO_HOST: Dst  (%zd, %zd, %zd)", pZ * localZ * (dimensions > 2? dimMgr.get_strides()[dim_manager_type::DimIdxZ]: 0),
                                                                    pY * localY * (dimensions > 1? dimMgr.get_strides()[dim_manager_type::DimIdxY]: 0),
                                                                    pX * localX);
                    DEBUG("TO_HOST: Src   (%zd, %zd, %zd)", pZ * (dimensions > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0),
                                                            pY * (dimensions > 1? gpuOffs_[dim_manager_type::DimIdxY]: 0),
                                                            pX *                  gpuOffs_[dim_manager_type::DimIdxX]);

                    myParms.srcPtr = make_cudaPitchedPtr(dataDev_ + blockOff,
                                                         sizeof(value_type) * hostInfo_->localDims_[dim_manager_type::DimIdxX],
                                                                              hostInfo_->localDims_[dim_manager_type::DimIdxX],
                                                         dimensions > 1? hostInfo_->localDims_[dim_manager_type::DimIdxY]: 1);

                    // We copy the whole chunk
                    myParms.srcPos = make_cudaPos(0, 0, 0);

                    myParms.dstPos = make_cudaPos(sizeof(value_type) * pX * localX,
                                                  pY * localY,
                                                  pZ * localZ);

                    // Only transfer the remaining elements
                    if (PartConf::Z)
                    localZ = std::min(localZ, array_index_t(dimMgr.dim_align(dim_manager_type::DimIdxZ) - pZ * localZ));
                    if (PartConf::Y)
                    localY = std::min(localY, array_index_t(dimMgr.dim_align(dim_manager_type::DimIdxY) - pY * localY));
                    if (PartConf::X)
                    localX = std::min(localX, array_index_t(dimMgr.dim_align(dim_manager_type::DimIdxX) - pX * localX));

                    if (localZ < 1 || localY < 1 || localZ < 1) continue;

                    DEBUG("TO_HOST: Extent: (%u %u %u)",
                          sizeof(value_type) * localX, localY, localZ);

                    myParms.extent = make_cudaExtent(sizeof(value_type) * localX,
                                                     localY,
                                                     localZ);

                    myParms.kind = cudaMemcpyDeviceToHost;

                    CUDA_CALL(cudaMemcpy3D(&myParms));
                }
            }
        }
    }

    __host__
    void to_device(host_storage_type &host)
    {
        TRACE_FUNCTION();

        value_type *unaligned = host.addr();
        auto &dimMgr = this->get_dim_manager();

        unsigned partZ = (dimensions > 2)? arrayPartitionGrid_[dim_manager_type::DimIdxZ]: 1;
        unsigned partY = (dimensions > 1)? arrayPartitionGrid_[dim_manager_type::DimIdxY]: 1;
        unsigned partX =                   arrayPartitionGrid_[dim_manager_type::DimIdxX];

        cudaMemcpy3DParms myParms;
        memset(&myParms, 0, sizeof(myParms));
        myParms.srcPtr = make_cudaPitchedPtr(unaligned,
                                             sizeof(value_type) * dimMgr.dim_align(dim_manager_type::DimIdxX),
                                                                  dimMgr.dim_align(dim_manager_type::DimIdxX),
                                             dimensions > 1? dimMgr.dim_align(dim_manager_type::DimIdxY): 1);

        for (unsigned pZ : utils::make_range(partZ)) {
            for (unsigned pY : utils::make_range(partY)) {
                for (unsigned pX : utils::make_range(partX)) {
                    array_index_t localZ = dimensions > 2? hostInfo_->localDims_[dim_manager_type::DimIdxZ]: 1;
                    array_index_t localY = dimensions > 1? hostInfo_->localDims_[dim_manager_type::DimIdxY]: 1;
                    array_index_t localX =                 hostInfo_->localDims_[dim_manager_type::DimIdxX];

                    array_index_t blockOff = pZ * (dimensions > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0) +
                                             pY * (dimensions > 1? gpuOffs_[dim_manager_type::DimIdxY]: 0) +
                                             pX *                  gpuOffs_[dim_manager_type::DimIdxX];

                    DEBUG("TO_DEVICE: Src Block Off: %u",blockOff);

                    DEBUG("TO_DEVICE: Block (%u, %u, %u)", pZ, pY, pX);
                    DEBUG("TO_DEVICE: Dst   (%zd, %zd, %zd)",
                          pZ * (dimensions > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0),
                          pY * (dimensions > 1? gpuOffs_[dim_manager_type::DimIdxY]: 0),
                          pX *                  gpuOffs_[dim_manager_type::DimIdxX]);
                    DEBUG("TO_DEVICE: Src   (%zd, %zd, %zd)",
                          pZ * localZ * (dimensions > 2? dimMgr.get_strides()[dim_manager_type::DimIdxZ]: 0),
                          pY * localY * (dimensions > 1? dimMgr.get_strides()[dim_manager_type::DimIdxY]: 0),
                          pX * localX);

                    myParms.dstPtr = make_cudaPitchedPtr(dataDev_ + blockOff,
                                                         sizeof(value_type) * hostInfo_->localDims_[dim_manager_type::DimIdxX],
                                                                              hostInfo_->localDims_[dim_manager_type::DimIdxX],
                                                         dimensions > 1?   hostInfo_->localDims_[dim_manager_type::DimIdxY]: 1);

                    // We copy the whole chunk
                    myParms.dstPos = make_cudaPos(0, 0, 0);

                    myParms.srcPos = make_cudaPos(sizeof(value_type) * pX * localX,
                                                  pY * localY,
                                                  pZ * localZ);
                    // TODO: Check if this is necessary. If so, use it in to_host and the rest of implementations
                    if (PartConf::Z && pZ * localZ >= dimMgr.dim_align(dim_manager_type::DimIdxZ))
                        continue;
                    if (PartConf::Y && pY * localY >= dimMgr.dim_align(dim_manager_type::DimIdxY))
                        continue;
                    if (PartConf::X && pX * localX >= dimMgr.dim_align(dim_manager_type::DimIdxX))
                        continue;

                    // Only transfer the remaining elements
                    if (PartConf::Z)
                    localZ = std::min(localZ, array_index_t(dimMgr.dim_align(dim_manager_type::DimIdxZ) - pZ * localZ));
                    if (PartConf::Y)
                    localY = std::min(localY, array_index_t(dimMgr.dim_align(dim_manager_type::DimIdxY) - pY * localY));
                    if (PartConf::X)
                    localX = std::min(localX, array_index_t(dimMgr.dim_align(dim_manager_type::DimIdxX) - pX * localX));

                    DEBUG("TO_DEVICE: Extent: (%u %u %u)",
                          sizeof(value_type) * localX, localY, localZ);

                    myParms.extent = make_cudaExtent(sizeof(value_type) * localX,
                                                     localY,
                                                     localZ);

                    myParms.kind = cudaMemcpyHostToDevice;

                    CUDA_CALL(cudaMemcpy3D(&myParms));
                }
            }
        }
    }

    unsigned get_ngpus() const
    {
        return hostInfo_->gpus;
    }

    template <typename... Idxs>
    __device__ inline
    value_type &access_pos(Idxs... idxs)
    {
        array_index_t idx;
        idx = indexer_type::access_pos(localOffs_, arrayPartitionGrid_,
                                       gpuOffs_,
                                       idxs...);
        return this->dataDev_[idx];
    }

    template <typename... Idxs>
    __device__ inline
    const value_type &access_pos(Idxs... idxs) const
    {
        array_index_t idx;
        idx = indexer_type::access_pos(localOffs_, arrayPartitionGrid_,
                                       gpuOffs_,
                                       idxs...);
        return this->dataDev_[idx];
    }

private:
    CUDARRAYS_TESTED(lib_storage_test, host_reshape_cyclic)
};

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
