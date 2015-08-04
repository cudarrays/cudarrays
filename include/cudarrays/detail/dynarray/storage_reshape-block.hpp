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
#ifndef CUDARRAYS_DETAIL_DYNARRAY_STORAGE_RESHAPE_BLOCK_HPP_
#define CUDARRAYS_DETAIL_DYNARRAY_STORAGE_RESHAPE_BLOCK_HPP_

#include "../../utils.hpp"

#include "base.hpp"
#include "helpers.hpp"

namespace cudarrays {

template <typename T, typename StorageTraits>
class dynarray_storage<T, storage_tag::RESHAPE_BLOCK, StorageTraits> :
    public dynarray_base<T, StorageTraits::dimensions>
{
    static constexpr unsigned dimensions = StorageTraits::dimensions;

    using PartConf = storage_part_dim_helper<StorageTraits::partition_value, dimensions>;

    using base_storage_type = dynarray_base<T, dimensions>;
    using  dim_manager_type = typename base_storage_type::dim_manager_type;

    using indexer_type = index_block<typename StorageTraits::offsets_seq,
                                     typename StorageTraits::partitioning_seq>;

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

        DEBUG("Reshape> ALLOCATE");
        // Iterate on all array grid dimensions
        for (unsigned pZ : utils::make_range(partZ)) {
            for (unsigned pY : utils::make_range(partY)) {
                for (unsigned pX : utils::make_range(partX)) {
                    // Compute the linear index of the array partition to be allocated
                    unsigned linear = pZ * partX * partY   + pY * partX           + pX;
                    // Compute the index of the GPU where the partition must be allocated
                    unsigned idx    = pZ * gpuDimForArrayZ + pY * gpuDimForArrayY + pX * gpuDimForArrayX;

                    DEBUG("Reshape> in: %u,%u,%u -> %u", pZ, pY, pX, idx);

                    unsigned gpu = (idx >= config::PEER_GPUS)? 0 : idx;
                    // Set the device where data is allocated
                    CUDA_CALL(cudaSetDevice(gpu));
                    // Perform memory allocation
                    T *tmp;
                    CUDA_CALL(cudaMalloc((void **) &tmp, hostInfo_->elemsLocal * sizeof(T)));
                    if (idx == 0) {
                        // Initialize the base address of the allocation
                        dataDev_ = tmp;
                    } else {
                        // Check that allocations are contiguous in the virtual address space
                        ASSERT(dataDev_ + linear * hostInfo_->elemsLocal == tmp);
                    }

                    DEBUG("Reshape> - allocated %p (%zd) in GPU %u", tmp, hostInfo_->elemsLocal * sizeof(T), gpu);
                }
            }
        }

        dataDev_ += this->get_dim_manager().offset();
    }

public:
    template <unsigned DimsComp>
    __host__
    void compute_distribution_internal(const compute_mapping<DimsComp, dimensions> &mapping)
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
        hostInfo_->arrayPartitionGrid = arrayPartitionGrid;
        // 3- Compute dimensions of each tile
        std::array<array_size_t, dimensions> localDims = helper_distribution_get_local_dims(dims, arrayPartitionGrid);
        utils::copy(localDims, localDims_);
        // 4- Compute local offsets for the indexing functions
        std::array<array_size_t, dimensions - 1> localOffs = helper_distribution_get_local_offs(localDims);
        utils::copy(localOffs, localOffs_);
        // 5- Compute elements of each tile
        array_size_t elemsLocal = helper_distribution_get_local_elems(localDims, config::CUDA_VM_ALIGN_ELEMS<T>());
        hostInfo_->elemsLocal = elemsLocal;
        // 6- Compute the inter-GPU array offsets for each dimension (iterate from lowest-order dimension)
        std::array<array_size_t, dimensions> gpuOffs = helper_distribution_get_intergpu_offs(elemsLocal, arrayPartitionGrid, arrayDimToCompDim);
        utils::copy(gpuOffs, gpuOffs_);
        // 7- Compute the GPU grid offsets (iterate from lowest-order dimension)
        std::array<unsigned, DimsComp> gpuGridOffs = helper_distribution_gpu_get_offs(gpuGrid);
        // 8- Compute the array to GPU mapping needed for the allocation based on the grid offsets
        std::array<unsigned, dimensions> arrayDimToGpus = helper_distribution_get_array_dim_to_gpus(gpuGridOffs, arrayDimToCompDim);
        utils::copy(arrayDimToGpus, hostInfo_->arrayDimToGpus);

        DEBUG("Reshape> BASE INFO");
        DEBUG("Reshape> - array dims: %s", utils::to_string(dims).c_str());
        DEBUG("Reshape> - comp  dims: %u", DimsComp);
        DEBUG("Reshape> - comp -> array: %s", utils::to_string(arrayDimToCompDim).c_str());

        DEBUG("Reshape> PARTITIONING");
        DEBUG("Reshape> - gpus: %u", mapping.comp.procs);
        DEBUG("Reshape> - comp  part: %s (%u)", utils::to_string(mapping.comp.info).c_str(), mapping.comp.get_part_dims());
        DEBUG("Reshape> - comp  grid: %s", utils::to_string(gpuGrid).c_str());
        DEBUG("Reshape> - array grid: %s", utils::to_string(hostInfo_->arrayPartitionGrid).c_str());
        DEBUG("Reshape> - local elems: %s (%zd)", utils::to_string(localDims_).c_str(), size_t(hostInfo_->elemsLocal));
        DEBUG("Reshape> - local offs: %s", utils::to_string(localOffs_).c_str());

        DEBUG("Reshape> - array grid offsets: %s", utils::to_string(hostInfo_->arrayDimToGpus).c_str());
        DEBUG("Reshape> - gpu   grid offsets: %s", utils::to_string(gpuOffs_).c_str());
    }

    template <unsigned DimsComp>
    __host__ void
    compute_distribution(const compute_mapping<DimsComp, dimensions> &mapping)
    {
        DEBUG("=========================");
        DEBUG("Reshape> DISTRIBUTE BEGIN");

        if (hostInfo_ != nullptr) delete hostInfo_;
        hostInfo_ = new storage_host_info(mapping.comp.procs);

        compute_distribution_internal(mapping);

        DEBUG("Reshape> DISTRIBUTE END");
        DEBUG("=========================");
    }

    template <unsigned DimsComp>
    __host__ bool
    distribute(const compute_mapping<DimsComp, dimensions> &mapping)
    {
        bool ret = false;

        // Only distribute the first time. Otherwise use redistribute
        if (!dataDev_) {
            DEBUG("=====================");
            DEBUG("Reshape> ALLOC: BEGIN");

            hostInfo_ = new storage_host_info(mapping.comp.procs);

            compute_distribution_internal(mapping);

            alloc();

            ret = true;

            DEBUG("Reshape> ALLOC: END");
            DEBUG("=====================");
        }

        return ret;
    }

    __host__ bool
    distribute(const std::vector<unsigned> &/*gpus*/)
    {
        return false;
    }

    __host__ bool
    is_distributed() const
    {
        return dataDev_ != nullptr;
    }

private:
    T *dataDev_;

    array_size_t localDims_[dimensions];
    array_size_t localOffs_[dimensions - 1];
    array_size_t gpuOffs_[dimensions];

    struct storage_host_info {
        unsigned gpus;

        array_size_t elemsLocal;
        std::array<unsigned, dimensions> arrayPartitionGrid;
        std::array<unsigned, dimensions> arrayDimToGpus;

        storage_host_info(unsigned _gpus) :
            gpus(_gpus)
        {
        }
    };

    storage_host_info *hostInfo_;

public:
    __host__
    dynarray_storage(const extents<dimensions> &ext,
                     const align_t &align) :
        base_storage_type(ext, align),
        dataDev_(nullptr),
        hostInfo_(nullptr)
    {
    }

    __host__
    virtual ~dynarray_storage()
    {
        if (dataDev_ != nullptr) {
            // Free device memory (1 chunk per GPU)
            for (unsigned idx = 0; idx < hostInfo_->gpus; ++idx) {
                DEBUG("Reshape> - freeing %p", dataDev_ - this->get_dim_manager().offset() + hostInfo_->elemsLocal * idx);
                CUDA_CALL(cudaFree(dataDev_ - this->get_dim_manager().offset() + hostInfo_->elemsLocal * idx));
            }
        }

        if (hostInfo_ != nullptr) {
            delete hostInfo_;
        }
    }

    __host__
    void to_host(host_storage &host)
    {
        T *unaligned = host.addr<T>();
        auto &dimMgr = this->get_dim_manager();

        unsigned partZ = (dimensions > 2)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxZ]: 1;
        unsigned partY = (dimensions > 1)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxY]: 1;
        unsigned partX =                   hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxX];

        cudaMemcpy3DParms myParms;
        memset(&myParms, 0, sizeof(myParms));
        myParms.dstPtr = make_cudaPitchedPtr(unaligned,
                                             sizeof(T) * dimMgr.dim_align(dim_manager_type::DimIdxX),
                                                         dimMgr.dim_align(dim_manager_type::DimIdxX),
                                             dimensions> 1? dimMgr.dim_align(dim_manager_type::DimIdxY): 1);

        for (unsigned pZ : utils::make_range(partZ)) {
            for (unsigned pY : utils::make_range(partY)) {
                for (unsigned pX : utils::make_range(partX)) {
                    array_index_t localZ = dimensions > 2? this->localDims_[dim_manager_type::DimIdxZ]: 1;
                    array_index_t localY = dimensions > 1? this->localDims_[dim_manager_type::DimIdxY]: 1;
                    array_index_t localX =                 this->localDims_[dim_manager_type::DimIdxX];

                    array_index_t blockOff = pZ * (dimensions > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0) +
                                             pY * (dimensions > 1? gpuOffs_[dim_manager_type::DimIdxY]: 0) +
                                             pX *                  gpuOffs_[dim_manager_type::DimIdxX];

                    DEBUG("TO_HOST: Extent: (%u %u %u)", sizeof(T) * localDims_[dim_manager_type::DimIdxX],
                                                         localY,
                                                         localZ);

                    DEBUG("TO_HOST: Src Block Off: %u", blockOff);

                    DEBUG("TO_HOST: Block (%u, %u, %u)", pZ, pY, pX);
                    DEBUG("TO_HOST: Dst  (%zd, %zd, %zd)", pZ * localZ * (dimensions > 2? dimMgr.get_strides()[dim_manager_type::DimIdxZ]: 0),
                                                           pY * localY * (dimensions > 1? dimMgr.get_strides()[dim_manager_type::DimIdxY]: 0),
                                                           pX * localX);
                    DEBUG("TO_HOST: Src   (%zd, %zd, %zd)", pZ * (dimensions > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0),
                                                            pY * (dimensions > 1? gpuOffs_[dim_manager_type::DimIdxY]: 0),
                                                            pX *                  gpuOffs_[dim_manager_type::DimIdxX]);

                    myParms.srcPtr = make_cudaPitchedPtr(dataDev_ + blockOff,
                                                         sizeof(T) * localDims_[dim_manager_type::DimIdxX],
                                                                     localDims_[dim_manager_type::DimIdxX],
                                                         dimensions > 1? localDims_[dim_manager_type::DimIdxY]: 1);

                    // We copy the whole chunk
                    myParms.srcPos = make_cudaPos(0, 0, 0);

                    myParms.dstPos = make_cudaPos(sizeof(T) * pX * localX,
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

                    myParms.extent = make_cudaExtent(sizeof(T) * localX,
                                                     localY,
                                                     localZ);

                    myParms.kind = cudaMemcpyDeviceToHost;

                    CUDA_CALL(cudaMemcpy3D(&myParms));
                }
            }
        }
    }

    __host__
    void to_device(host_storage &host)
    {
        T *unaligned = host.addr<T>();
        auto &dimMgr = this->get_dim_manager();

        unsigned partZ = (dimensions > 2)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxZ]: 1;
        unsigned partY = (dimensions > 1)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxY]: 1;
        unsigned partX =                   hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxX];

        cudaMemcpy3DParms myParms;
        memset(&myParms, 0, sizeof(myParms));
        myParms.srcPtr = make_cudaPitchedPtr(unaligned,
                                             sizeof(T) * dimMgr.dim_align(dim_manager_type::DimIdxX),
                                                         dimMgr.dim_align(dim_manager_type::DimIdxX),
                                             dimensions > 1? dimMgr.dim_align(dim_manager_type::DimIdxY): 1);

        for (unsigned pZ : utils::make_range(partZ)) {
            for (unsigned pY : utils::make_range(partY)) {
                for (unsigned pX : utils::make_range(partX)) {
                    array_index_t localZ = dimensions > 2? this->localDims_[dim_manager_type::DimIdxZ]: 1;
                    array_index_t localY = dimensions > 1? this->localDims_[dim_manager_type::DimIdxY]: 1;
                    array_index_t localX =                   this->localDims_[dim_manager_type::DimIdxX];

                    array_index_t blockOff = pZ * (dimensions > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0) +
                                             pY * (dimensions > 1? gpuOffs_[dim_manager_type::DimIdxY]: 0) +
                                             pX *                  gpuOffs_[dim_manager_type::DimIdxX];

                    DEBUG("TO_DEVICE: Src Block Off: %u", blockOff);

                    DEBUG("TO_DEVICE: Block (%u, %u, %u)", pZ, pY, pX);
                    DEBUG("TO_DEVICE: Dst   (%zd, %zd, %zd)", pZ * (dimensions > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0),
                                                              pY * (dimensions > 1? gpuOffs_[dim_manager_type::DimIdxY]: 0),
                                                              pX *                  gpuOffs_[dim_manager_type::DimIdxX]);
                    DEBUG("TO_DEVICE: Src   (%zd, %zd, %zd)", pZ * localZ * (dimensions > 2? dimMgr.get_strides()[dim_manager_type::DimIdxZ]: 0),
                                                              pY * localY * (dimensions > 1? dimMgr.get_strides()[dim_manager_type::DimIdxY]: 0),
                                                              pX * localX);

                    myParms.dstPtr = make_cudaPitchedPtr(dataDev_ + blockOff,
                                                         sizeof(T) * localDims_[dim_manager_type::DimIdxX],
                                                                     localDims_[dim_manager_type::DimIdxX],
                                                         dimensions > 1? localDims_[dim_manager_type::DimIdxY]: 1);

                    // We copy the whole chunk
                    myParms.dstPos = make_cudaPos(0, 0, 0);

                    myParms.srcPos = make_cudaPos(sizeof(T) * pX * localX,
                                                  pY * localY,
                                                  pZ * localZ);

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

                    DEBUG("TO_DEVICE: Extent: (%u %u %u)", sizeof(T) * localX,
                                                           localY,
                                                           localZ);

                    myParms.extent = make_cudaExtent(sizeof(T) * localX,
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
    T &access_pos(Idxs... idxs)
    {
        array_index_t idx = indexer_type::access_pos(localOffs_, localDims_,
                                                     gpuOffs_,
                                                     idxs...);
        return this->dataDev_[idx];
    }

    template <typename... Idxs>
    __device__ inline
    const T &access_pos(Idxs... idxs) const
    {
        array_index_t idx = indexer_type::access_pos(localOffs_, localDims_,
                                                     gpuOffs_,
                                                     idxs...);
        return this->dataDev_[idx];
    }

private:
    CUDARRAYS_TESTED(lib_storage_test, host_reshape_block)
};

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
