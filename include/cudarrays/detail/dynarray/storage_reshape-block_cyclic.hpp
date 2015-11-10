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
#ifndef CUDARRAYS_DETAIL_DYNARRAY_STORAGE_RESHAPE_BLOCK_CYCLIC_HPP_
#define CUDARRAYS_DETAIL_DYNARRAY_STORAGE_RESHAPE_BLOCK_CYCLIC_HPP_

#include "../../utils.hpp"

#include "base.hpp"
#include "helpers.hpp"

namespace cudarrays {

template <typename StorageTraits>
class dynarray_storage<detail::storage_tag::RESHAPE_BLOCK_CYCLIC, StorageTraits> :
    public dynarray_base<StorageTraits>
{
    static constexpr array_size_t BlockSize = 1;

    using base_storage_type = dynarray_base<StorageTraits>;
    using        value_type = typename base_storage_type::value_type;
    using    alignment_type = typename base_storage_type::alignment_type;
    using  dim_manager_type = typename base_storage_type::dim_manager_type;
    using host_storage_type = typename base_storage_type::host_storage_type;

    static constexpr auto dimensions = base_storage_type::dimensions;

    using PartConf = storage_part_dim_helper<StorageTraits::partition_value, dimensions>;

    using indexer_type = index_block_cyclic<typename StorageTraits::offsets_seq,
                                            typename StorageTraits::partitioning_seq,
                                            BlockSize>;

    __host__
    void alloc()
    {
        // Array grid layout
        unsigned partZ = (dimensions > 2)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxZ]: 1;
        unsigned partY = (dimensions > 1)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxY]: 1;
        unsigned partX =                   hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxX];
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
    __host__
    void compute_distribution_internal_single(const cudarrays::compute_mapping<DimsComp, dimensions> &mapping)
    {
        // Compute the number of partitioned dimensions
        for (unsigned i = 0; i < dimensions; ++i) {
            hostInfo_->mapping[i] = DimsComp - (mapping.info[i] + 1);
        }
        DEBUG("ALLOC: mapping: %s", hostInfo_->mapping);

        // Count partitioned dimensions in array and computation
        unsigned arrayPartDims = mapping.get_array_part_dims();
        unsigned  compPartDims = utils::count(mapping.comp.info, true);

        if (arrayPartDims > compPartDims)
            FATAL("Not enough partitioned comp dims: %u, to partition %u array dims", compPartDims, arrayPartDims);

        // Distribute the partition uniformly across GPUs
        hostInfo_->gpus = mapping.comp.procs;

        hostInfo_->elemsLocal = 1;

        // Create the GPU grid
        for (unsigned i = 0; i < DimsComp; ++i) {
            hostInfo_->gpuGrid[i] = 1;
        }
        // Compute the
        for (unsigned i = 0; i < dimensions; ++i) {
            gridDims_[i] = 1;
            fakeBlockDims_[i] = 1;
        }

        // Compute the array grid and the local sizes
        for (unsigned i = 0; i < dimensions; ++i) {
            hostInfo_->localDims_[i] = this->get_dim_manager().dims()[i];
            blockDims_[i]            = this->get_dim_manager().dims()[i];

            hostInfo_->arrayPartitionGrid[i] = 1;
            hostInfo_->elemsLocal *= hostInfo_->localDims_[i];
        }

        // Adjust to VM SIZE
        static const array_size_t CUDA_VM_ALIGN_ELEMS = system::CUDA_VM_ALIGN/sizeof(value_type);
        hostInfo_->elemsLocal = utils::round_next(hostInfo_->elemsLocal, CUDA_VM_ALIGN_ELEMS);

        array_size_t prevLocalOff = 1;
        // Compute the inter-GPU offsets for each dimension
        for (unsigned i = 0; i < dimensions; ++i) {
            unsigned dim = dimensions - (i + 1);
            gpuOffs_[dim] = 0;

            if (i > 0) {
                prevLocalOff *= hostInfo_->localDims_[dim + 1];
                localOffs_[dim] = prevLocalOff;
            }
        }

        // Compute the array to GPU mapping needed for the allocation based on the grid offsets
        for (unsigned i = 0; i < dimensions; ++i) {
            hostInfo_->arrayDimToGpus[i] = 0;
        }

        DEBUG("ALLOC: gpus: %u", mapping.comp.procs);
        DEBUG("ALLOC: comp  part: %s (%u)", mapping.comp.info, compPartDims);
        DEBUG("ALLOC: comp  grid: %s", hostInfo_->gpuGrid.get());
        DEBUG("ALLOC: array grid: %s", hostInfo_->arrayPartitionGrid);
        DEBUG("ALLOC: local elems: %s (%zd)", hostInfo_->localDims_, size_t(hostInfo_->elemsLocal));
        DEBUG("ALLOC: local offs: %s", localOffs_);

        DEBUG("ALLOC: grid offsets: %s", hostInfo_->arrayDimToGpus);
        DEBUG("ALLOC: gpu offsets: %s", gpuOffs_);
    }

    template <unsigned DimsComp>
    __host__
    void compute_distribution_internal(const cudarrays::compute_mapping<DimsComp, dimensions> &mapping)
    {
        // Compute the number of partitioned dimensions
        for (unsigned i = 0; i < dimensions; ++i) {
            hostInfo_->mapping[i] = DimsComp - (mapping.info[i] + 1);
        }
        DEBUG("ALLOC: mapping: %s", hostInfo_->mapping);

        // Count partitioned dimensions in array and computation
        unsigned arrayPartDims = mapping.get_array_part_dims();
        unsigned  compPartDims = utils::count(mapping.comp.info, true);

        if (arrayPartDims > compPartDims)
            FATAL("Not enough partitioned comp dims: %u, to partition %u array dims", compPartDims, arrayPartDims);

        // Check the minumum number of partitions
        if ((1u << compPartDims) > mapping.comp.procs)
            FATAL("Not enough GPUs (%u), to partition %u", mapping.comp.procs, arrayPartDims);

        // Distribute the partition uniformly across GPUs
        hostInfo_->gpus = mapping.comp.procs;

        // Check if we can map the partitions on the GPUs
        std::vector<unsigned> factorsGPUs = utils::get_factors(hostInfo_->gpus);
        std::sort(factorsGPUs.begin(), factorsGPUs.end(), std::greater<unsigned>());

        if (factorsGPUs.size() < compPartDims)
            FATAL("CUDArrays cannot partition %u dimensions into %u GPUs", arrayPartDims, mapping.comp.procs);

        hostInfo_->elemsLocal = 1;

        // Create the GPU grid
        unsigned j = 0;
        for (unsigned i = 0; i < DimsComp; ++i) {
            unsigned partition = 1;
            if (mapping.comp.info[i]) {
                ASSERT(j < factorsGPUs.size());

                std::vector<unsigned>::iterator pos = factorsGPUs.begin() + j;
                size_t inc = (j == 0)? factorsGPUs.size() - compPartDims + 1: 1;

                // DEBUG("ALLOC: Collapsing%u: %zd:%zd", i, j, j + inc);
                partition = std::accumulate(pos, pos + inc, 1, std::multiplies<unsigned>());
                j += inc;
            }
            hostInfo_->gpuGrid[i] = partition;
        }

        // Compute the array grid and the local sizes
        for (unsigned i = 0; i < dimensions; ++i) {
            int compDim = hostInfo_->mapping[i];

            unsigned partition = 1;
            if (compDim != DimsComp) {
                DEBUG("ALLOC: mapping array dim %d on comp dim %d", i, compDim);
                partition = hostInfo_->gpuGrid[compDim];
            } else {
                // TODO: REPLICATION
            }

            hostInfo_->localDims_[i] = utils::div_ceil(this->get_dim_manager().dims()[i], partition);

            hostInfo_->arrayPartitionGrid[i] = partition;
            hostInfo_->elemsLocal *= hostInfo_->localDims_[i];
        }

        // Adjust to VM SIZE
        array_size_t CUDA_VM_ALIGN_ELEMS = system::CUDA_VM_ALIGN/sizeof(value_type);
        hostInfo_->elemsLocal = utils::round_next(hostInfo_->elemsLocal, CUDA_VM_ALIGN_ELEMS);

        array_size_t off = 1;
        array_size_t prevLocalOff = 1;
        // Compute the inter-GPU offsets for each dimension
        for (unsigned i = 0; i < dimensions; ++i) {
            unsigned dim = dimensions - (i + 1);
            if (mapping.info[dim] != -1) {
                gpuOffs_[dim] = off * hostInfo_->elemsLocal;
                off *= hostInfo_->arrayPartitionGrid[dim];
            } else {
                gpuOffs_[dim] = 0;
            }

            if (i > 0) {
                prevLocalOff *= hostInfo_->localDims_[dim + 1];
                localOffs_[dim] = prevLocalOff;
            }
        }

        unsigned arrayDimToGpus[dimensions];
        unsigned gridOff = 1;
        // Compute the local grid offsets
        for (unsigned i = 0; i < dimensions; ++i) {
            unsigned dim = dimensions - (i + 1);

            if (dim < DimsComp) {
                arrayDimToGpus[dim] = gridOff;
                gridOff *= hostInfo_->gpuGrid[dim];
            } else {
                arrayDimToGpus[dim] = gridOff;
            }
        }

        // Compute the array to GPU mapping needed for the allocation based on the grid offsets
        for (unsigned i = 0; i < dimensions; ++i) {
            int j = hostInfo_->mapping[i];
            if (j != DimsComp) {
                hostInfo_->arrayDimToGpus[i] = arrayDimToGpus[j];
            } else
                hostInfo_->arrayDimToGpus[i] = 0;
        }

        // DEBUG("ALLOC: factors: %s", utils::to_string(factorsGPUs));
        DEBUG("ALLOC: gpus: %u", mapping.comp.procs);
        DEBUG("ALLOC: comp  part: %s (%u)", mapping.comp.info, compPartDims);
        DEBUG("ALLOC: comp  grid: %s", hostInfo_->gpuGrid.get());
        DEBUG("ALLOC: array grid: %s", hostInfo_->arrayPartitionGrid);
        DEBUG("ALLOC: local elems: %s (%zd)", hostInfo_->localDims_, size_t(hostInfo_->elemsLocal));
        DEBUG("ALLOC: local offs: %s", localOffs_);

        DEBUG("ALLOC: grid offsets: %s", hostInfo_->arrayDimToGpus);
        DEBUG("ALLOC: gpu offsets: %s", gpuOffs_);
    }

    template <unsigned DimsComp>
    __host__ void
    compute_distribution(const cudarrays::compute_mapping<DimsComp, dimensions> &mapping)
    {
        hostInfo_.reset(new storage_host_info(mapping.comp.info));

        compute_distribution_internal(mapping);
    }

    template <unsigned DimsComp>
    __host__ bool
    distribute(const cudarrays::compute_mapping<DimsComp, dimensions> &mapping)
    {
        if (!dataDev_) {
            hostInfo_.reset(new storage_host_info(mapping.comp.info));

            if (mapping.comp.procs == 1)
                compute_distribution_internal_single(mapping);
            else {
                FATAL("Not implemented yet");
                compute_distribution_internal(mapping);
            }
            alloc();

            return true;
        }
        return false;
    }

    __host__ bool
    distribute(const std::vector<unsigned> &)
    {
        return false;
    }

    __host__ bool
    is_distributed() const
    {
        return dataDev_ != NULL;
    }

private:
    value_type *dataDev_;

    array_size_t gridDims_[dimensions];
    array_size_t blockDims_[dimensions];
    array_size_t fakeBlockDims_[dimensions];
    array_size_t localOffs_[dimensions - 1];
    array_size_t gpuOffs_[dimensions];

    struct storage_host_info {
        array_size_t elemsLocal;
        array_size_t localDims_[dimensions];

        unsigned gpus;
        unsigned arrayPartitionGrid[dimensions];
        unsigned arrayDimToGpus[dimensions];
        int mapping[dimensions];

        std::unique_ptr<unsigned[]> gpuGrid;
        unsigned compDims;

        template <size_t DimsComp>
        __host__
        storage_host_info(const std::array<bool, DimsComp> &) :
            elemsLocal{0},
            gpus{0},
            gpuGrid{new unsigned[DimsComp]},
            compDims{DimsComp}
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
            for (unsigned idx : utils::make_range(hostInfo_->gpus)) {
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

        unsigned partZ = (dimensions > 2)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxZ]: 1;
        unsigned partY = (dimensions > 1)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxY]: 1;
        unsigned partX =                   hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxX];

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

                    DEBUG("TO_HOST: Extent: (%u %u %u)", sizeof(value_type) * hostInfo_->localDims_[dim_manager_type::DimIdxX],
                                                         localY,
                                                         localZ);

                    DEBUG("TO_HOST: Src Block Off: %u", blockOff);

                    DEBUG("TO_HOST: Block (%u, %u, %u)", pZ, pY, pX);
                    DEBUG("TO_HOST: Dst  (%zd, %zd, %zd)",
                          pZ * localZ * (dimensions > 2? dimMgr.get_strides()[dim_manager_type::DimIdxZ]: 0),
                          pY * localY * (dimensions > 1? dimMgr.get_strides()[dim_manager_type::DimIdxY]: 0),
                          pX * localX);
                    DEBUG("TO_HOST: Src   (%zd, %zd, %zd)",
                          pZ * (dimensions > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0),
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

        unsigned partZ = (dimensions > 2)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxZ]: 1;
        unsigned partY = (dimensions > 1)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxY]: 1;
        unsigned partX =                   hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxX];

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

                    DEBUG("TO_DEVICE: Src Block Off: %u", blockOff);

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

                    if (PartConf::Z && pZ * localZ >= dimMgr.dims_align()[dim_manager_type::DimIdxZ])
                        continue;
                    if (PartConf::Y && pY * localY >= dimMgr.dims_align()[dim_manager_type::DimIdxY])
                        continue;
                    if (PartConf::X && pX * localX >= dimMgr.dims_align()[dim_manager_type::DimIdxX])
                        continue;

                    // Only transfer the remaining elements
                    if (PartConf::Z)
                    localZ = std::min(localZ, array_index_t(dimMgr.dims_align()[dim_manager_type::DimIdxZ] - pZ * localZ));
                    if (PartConf::Y)
                    localY = std::min(localY, array_index_t(dimMgr.dims_align()[dim_manager_type::DimIdxY] - pY * localY));
                    if (PartConf::X)
                    localX = std::min(localX, array_index_t(dimMgr.dims_align()[dim_manager_type::DimIdxX] - pX * localX));

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
        idx = indexer_type::access_pos(localOffs_, blockDims_, fakeBlockDims_,
                                       gridDims_,
                                       gpuOffs_,
                                       idxs...);
        return this->dataDev_[idx];
    }

    template <typename... Idxs>
    __device__ inline
    const value_type &access_pos(Idxs... idxs) const
    {
        array_index_t idx;
        idx = indexer_type::access_pos(localOffs_, blockDims_, fakeBlockDims_,
                                       gridDims_,
                                       gpuOffs_,
                                       idxs...);
        return this->dataDev_[idx];
    }

private:
    CUDARRAYS_TESTED(lib_storage_test, host_reshape_block_cyclic)
};

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
