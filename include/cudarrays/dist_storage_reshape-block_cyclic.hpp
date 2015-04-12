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
#ifndef CUDARRAYS_DIST_STORAGE_RESHAPE_BLOCK_CYCLIC_HPP_
#define CUDARRAYS_DIST_STORAGE_RESHAPE_BLOCK_CYCLIC_HPP_

#include "log.hpp"
#include "storage.hpp"

namespace cudarrays {

template <typename T, unsigned Dims, typename PartConf>
class array_storage<T, Dims, RESHAPE_BLOCK_CYCLIC, PartConf> :
    public array_storage_base<T, Dims>
{
    static constexpr array_size_t BlockSize = 1;

    using base_storage_type = array_storage_base<T, Dims>;
    using  dim_manager_type = typename base_storage_type::dim_manager_type;
    using      extents_type = typename base_storage_type::extents_type;

    __host__
    void alloc()
    {
        // Only support offsets in 1D arrays
        assert(this->get_dim_manager().get_offset() == 0 || Dims != 1);

        CUDA_CALL(cudaSetDevice(0));

        // Array grid layout
        unsigned partZ = (Dims > 2)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxZ]: 1;
        unsigned partY = (Dims > 1)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxY]: 1;
        unsigned partX =             hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxX];
        // Array-to-GPU translation
        unsigned gpuDimForArrayZ = (Dims > 2)? hostInfo_->arrayDimToGpus[dim_manager_type::DimIdxZ]: 1;
        unsigned gpuDimForArrayY = (Dims > 1)? hostInfo_->arrayDimToGpus[dim_manager_type::DimIdxY]: 1;
        unsigned gpuDimForArrayX =             hostInfo_->arrayDimToGpus[dim_manager_type::DimIdxX];

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

                    unsigned gpu = (idx >= PEER_GPUS)? 0 : idx;
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

        dataDev_ += this->get_dim_manager().get_offset();
    }

public:
    template <unsigned DimsComp>
    __host__
    void compute_distribution_internal_single(const compute_mapping<DimsComp, Dims> &mapping)
    {
        // Compute the number of partitioned dimensions
        for (unsigned i = 0; i < Dims; ++i) {
            hostInfo_->mapping[i] = DimsComp - (mapping.info[i] + 1);
        }
        DEBUG("Reshape> ALLOC: mapping: %s", to_string(hostInfo_->mapping, Dims).c_str());

        // Count partitioned dimensions in array and computation
        unsigned arrayPartDims = mapping.get_array_part_dims();
        unsigned  compPartDims = count(mapping.comp.info, true);

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
        for (unsigned i = 0; i < Dims; ++i) {
            gridDims_[i] = 1;
            fakeBlockDims_[i] = 1;
        }

        // Compute the array grid and the local sizes
        for (unsigned i = 0; i < Dims; ++i) {
            hostInfo_->localDims_[i] = this->get_dim_manager().sizes_[i];
            blockDims_[i]            = this->get_dim_manager().sizes_[i];

            hostInfo_->arrayPartitionGrid[i] = 1;
            hostInfo_->elemsLocal *= hostInfo_->localDims_[i];
        }

        // Adjust to VM SIZE
        static const array_size_t CUDA_VM_ALIGN_ELEMS = CUDA_VM_ALIGN/sizeof(T);
        hostInfo_->elemsLocal = round_next(hostInfo_->elemsLocal, CUDA_VM_ALIGN_ELEMS);

        array_size_t prevLocalOff = 1;
        // Compute the inter-GPU offsets for each dimension
        for (unsigned i = 0; i < Dims; ++i) {
            unsigned dim = Dims - (i + 1);
            gpuOffs_[dim] = 0;

            if (i > 0) {
                prevLocalOff *= hostInfo_->localDims_[dim + 1];
                localOffs_[dim] = prevLocalOff;
            }
        }

        // Compute the array to GPU mapping needed for the allocation based on the grid offsets
        for (unsigned i = 0; i < Dims; ++i) {
            hostInfo_->arrayDimToGpus[i] = 0;
        }

        DEBUG("Reshape> ALLOC: gpus: %u", mapping.comp.procs);
        DEBUG("Reshape> ALLOC: comp  part: %s (%u)", to_string(mapping.comp.info).c_str(), compPartDims);
        DEBUG("Reshape> ALLOC: comp  grid: %s", to_string(hostInfo_->gpuGrid, hostInfo_->compDims).c_str());
        DEBUG("Reshape> ALLOC: array grid: %s", to_string(hostInfo_->arrayPartitionGrid, Dims).c_str());
        DEBUG("Reshape> ALLOC: local elems: %s (%zd)", to_string(hostInfo_->localDims_, Dims).c_str(), size_t(hostInfo_->elemsLocal));
        DEBUG("Reshape> ALLOC: local offs: %s", to_string(localOffs_, Dims - 1).c_str());

        DEBUG("Reshape> ALLOC: grid offsets: %s", to_string(hostInfo_->arrayDimToGpus, Dims).c_str());
        DEBUG("Reshape> ALLOC: gpu offsets: %s", to_string(gpuOffs_, Dims).c_str());
    }

    template <unsigned DimsComp>
    __host__
    void compute_distribution_internal(const compute_mapping<DimsComp, Dims> &mapping)
    {
        // Compute the number of partitioned dimensions
        for (unsigned i = 0; i < Dims; ++i) {
            hostInfo_->mapping[i] = DimsComp - (mapping.info[i] + 1);
        }
        DEBUG("Reshape> ALLOC: mapping: %s", to_string(hostInfo_->mapping, Dims).c_str());

        // Count partitioned dimensions in array and computation
        unsigned arrayPartDims = mapping.get_array_part_dims();
        unsigned  compPartDims = count(mapping.comp.info, true);

        if (arrayPartDims > compPartDims)
            FATAL("Not enough partitioned comp dims: %u, to partition %u array dims", compPartDims, arrayPartDims);

        // Check the minumum number of partitions
        if ((1 << compPartDims) > mapping.comp.procs)
            FATAL("Not enough GPUs (%u), to partition %u", mapping.comp.procs, arrayPartDims);

        // Distribute the partition uniformly across GPUs
        hostInfo_->gpus = mapping.comp.procs;

        // Check if we can map the partitions on the GPUs
        std::vector<unsigned> factorsGPUs = get_factors(hostInfo_->gpus);
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

                // DEBUG("Reshape> ALLOC: Collapsing%u: %zd:%zd", i, j, j + inc);
                partition = std::accumulate(pos, pos + inc, 1, std::multiplies<unsigned>());
                j += inc;
            }
            hostInfo_->gpuGrid[i] = partition;
        }

        // Compute the array grid and the local sizes
        for (unsigned i = 0; i < Dims; ++i) {
            int compDim = hostInfo_->mapping[i];

            unsigned partition = 1;
            if (compDim != DimsComp) {
                DEBUG("Reshape> ALLOC: mapping array dim %d on comp dim %d", i, compDim);
                partition = hostInfo_->gpuGrid[compDim];
            } else {
                // TODO: REPLICATION
            }

            hostInfo_->localDims_[i] = div_ceil(this->get_dim_manager().sizes_[i], partition);

            hostInfo_->arrayPartitionGrid[i] = partition;
            hostInfo_->elemsLocal *= hostInfo_->localDims_[i];
        }

        // Adjust to VM SIZE
        array_size_t CUDA_VM_ALIGN_ELEMS = CUDA_VM_ALIGN/sizeof(T);
        hostInfo_->elemsLocal = round_next(hostInfo_->elemsLocal, CUDA_VM_ALIGN_ELEMS);

        array_size_t off = 1;
        array_size_t prevLocalOff = 1;
        // Compute the inter-GPU offsets for each dimension
        for (unsigned i = 0; i < Dims; ++i) {
            unsigned dim = Dims - (i + 1);
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

        unsigned arrayDimToGpus[Dims];
        unsigned gridOff = 1;
        // Compute the local grid offsets
        for (unsigned i = 0; i < Dims; ++i) {
            unsigned dim = Dims - (i + 1);

            if (dim < DimsComp) {
                arrayDimToGpus[dim] = gridOff;
                gridOff *= hostInfo_->gpuGrid[dim];
            } else {
                arrayDimToGpus[dim] = gridOff;
            }
        }

        // Compute the array to GPU mapping needed for the allocation based on the grid offsets
        for (unsigned i = 0; i < Dims; ++i) {
            int j = hostInfo_->mapping[i];
            if (j != DimsComp) {
                hostInfo_->arrayDimToGpus[i] = arrayDimToGpus[j];
            } else
                hostInfo_->arrayDimToGpus[i] = 0;
        }

        // DEBUG("Reshape> ALLOC: factors: %s", to_string(factorsGPUs).c_str());
        DEBUG("Reshape> ALLOC: gpus: %u", mapping.comp.procs);
        DEBUG("Reshape> ALLOC: comp  part: %s (%u)", to_string(mapping.comp.info).c_str(), compPartDims);
        DEBUG("Reshape> ALLOC: comp  grid: %s", to_string(hostInfo_->gpuGrid, hostInfo_->compDims).c_str());
        DEBUG("Reshape> ALLOC: array grid: %s", to_string(hostInfo_->arrayPartitionGrid, Dims).c_str());
        DEBUG("Reshape> ALLOC: local elems: %s (%zd)", to_string(hostInfo_->localDims_, Dims).c_str(), size_t(hostInfo_->elemsLocal));
        DEBUG("Reshape> ALLOC: local offs: %s", to_string(localOffs_, Dims - 1).c_str());

        DEBUG("Reshape> ALLOC: grid offsets: %s", to_string(hostInfo_->arrayDimToGpus, Dims).c_str());
        DEBUG("Reshape> ALLOC: gpu offsets: %s", to_string(gpuOffs_, Dims).c_str());
    }

    template <unsigned DimsComp>
    __host__ void
    compute_distribution(const compute_mapping<DimsComp, Dims> &mapping)
    {
        if (hostInfo_ != NULL) delete hostInfo_;
        hostInfo_ = new storage_host_info(mapping.comp.info);

        compute_distribution_internal(mapping);
    }

    template <unsigned DimsComp>
    __host__ bool
    distribute(const compute_mapping<DimsComp, Dims> &mapping)
    {
        if (!dataDev_) {
            hostInfo_ = new storage_host_info(mapping.comp.info);

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
    is_distributed()
    {
        return dataDev_ != NULL;
    }

private:
    T *dataDev_;

    array_size_t gridDims_[Dims];
    array_size_t blockDims_[Dims];
    array_size_t fakeBlockDims_[Dims];
    array_size_t localOffs_[Dims - 1];
    array_size_t gpuOffs_[Dims];

    struct storage_host_info {
        array_size_t elemsLocal;
        array_size_t localDims_[Dims];

        unsigned gpus;
        unsigned arrayPartitionGrid[Dims];
        unsigned arrayDimToGpus[Dims];
        int mapping[Dims];

        unsigned *gpuGrid;
        unsigned compDims;
        T **dataDevPtrs_;

        template <size_t DimsComp>
        __host__
        storage_host_info(const std::array<bool, DimsComp> &info)
        {
            gpuGrid = new unsigned[DimsComp];
            compDims = DimsComp;
        }

        ~storage_host_info()
        {
            delete []gpuGrid;
        }
    };

    storage_host_info *hostInfo_;

public:
    __host__
    array_storage(const extents_type &extents,
                  const align_t &align) :
        base_storage_type(extents, align),
        dataDev_(nullptr),
        hostInfo_(nullptr)
    {
    }

    __host__
    virtual ~array_storage()
    {
        if (dataDev_ != nullptr) {
            // Free device memory (1 chunk per GPU)
            for (unsigned idx = 0; idx < hostInfo_->gpus; ++idx) {
                DEBUG("Reshape> - freeing %p", dataDev_ - this->get_dim_manager().get_offset() + hostInfo_->elemsLocal * idx);
                CUDA_CALL(cudaFree(dataDev_ - this->get_dim_manager().get_offset() + hostInfo_->elemsLocal * idx));
            }
        }

        if (hostInfo_ != nullptr) {
            delete hostInfo_;
        }
    }

    __host__
    void to_host()
    {
        T *unaligned = this->get_host_storage().get_addr();

        unsigned partZ = (Dims > 2)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxZ]: 1;
        unsigned partY = (Dims > 1)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxY]: 1;
        unsigned partX =             hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxX];

        cudaMemcpy3DParms myParms = {0};
        myParms.dstPtr = make_cudaPitchedPtr(unaligned,
                                             sizeof(T) * this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxX],
                                                         this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxX],
                                             Dims > 1?   this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxY]: 1);

        for (unsigned pZ : utils::make_range(partZ)) {
            for (unsigned pY : utils::make_range(partY)) {
                for (unsigned pX : utils::make_range(partX)) {
                    array_index_t localZ = Dims > 2? hostInfo_->localDims_[dim_manager_type::DimIdxZ]: 1;
                    array_index_t localY = Dims > 1? hostInfo_->localDims_[dim_manager_type::DimIdxY]: 1;
                    array_index_t localX =           hostInfo_->localDims_[dim_manager_type::DimIdxX];

                    array_index_t blockOff = pZ * (Dims > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0) +
                                             pY * (Dims > 1? gpuOffs_[dim_manager_type::DimIdxY]: 0) +
                                             pX *            gpuOffs_[dim_manager_type::DimIdxX];

                    DEBUG("TO_HOST: Extent: (%u %u %u)", sizeof(T) * hostInfo_->localDims_[dim_manager_type::DimIdxX],
                                                         localY,
                                                         localZ);

                    DEBUG("TO_HOST: Src Block Off: %u", blockOff);

                    DEBUG("TO_HOST: Block (%u, %u, %u)", pZ, pY, pX);
                    DEBUG("TO_HOST: Dst  (%zd, %zd, %zd)", pZ * localZ * (Dims > 2? this->get_dim_manager().get_offs_align()[dim_manager_type::DimIdxZ]: 0),
                                                           pY * localY * (Dims > 1? this->get_dim_manager().get_offs_align()[dim_manager_type::DimIdxY]: 0),
                                                           pX * localX);
                    DEBUG("TO_HOST: Src   (%zd, %zd, %zd)", pZ * (Dims > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0),
                                                            pY * (Dims > 1? gpuOffs_[dim_manager_type::DimIdxY]: 0),
                                                            pX *            gpuOffs_[dim_manager_type::DimIdxX]);

                    myParms.srcPtr = make_cudaPitchedPtr(dataDev_ + blockOff,
                                                         sizeof(T) * hostInfo_->localDims_[dim_manager_type::DimIdxX],
                                                                     hostInfo_->localDims_[dim_manager_type::DimIdxX],
                                                         Dims > 1?   hostInfo_->localDims_[dim_manager_type::DimIdxY]: 1);

                    // We copy the whole chunk
                    myParms.srcPos = make_cudaPos(0, 0, 0);

                    myParms.dstPos = make_cudaPos(sizeof(T) * pX * localX,
                                                  pY * localY,
                                                  pZ * localZ);

                    // Only transfer the remaining elements
                    if (PartConf::Z)
                    localZ = std::min(localZ, array_index_t(this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxZ] - pZ * localZ));
                    if (PartConf::Y)
                    localY = std::min(localY, array_index_t(this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxY] - pY * localY));
                    if (PartConf::X)
                    localX = std::min(localX, array_index_t(this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxX] - pX * localX));

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
    void to_device()
    {
        T *unaligned = this->get_host_storage().get_addr();

        unsigned partZ = (Dims > 2)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxZ]: 1;
        unsigned partY = (Dims > 1)? hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxY]: 1;
        unsigned partX =             hostInfo_->arrayPartitionGrid[dim_manager_type::DimIdxX];

        cudaMemcpy3DParms myParms = {0};
        myParms.srcPtr = make_cudaPitchedPtr(unaligned,
                                             sizeof(T) * this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxX],
                                                         this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxX],
                                             Dims > 1?   this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxY]: 1);

        for (unsigned pZ : utils::make_range(partZ)) {
            for (unsigned pY : utils::make_range(partY)) {
                for (unsigned pX : utils::make_range(partX)) {
                    array_index_t localZ = Dims > 2? hostInfo_->localDims_[dim_manager_type::DimIdxZ]: 1;
                    array_index_t localY = Dims > 1? hostInfo_->localDims_[dim_manager_type::DimIdxY]: 1;
                    array_index_t localX =           hostInfo_->localDims_[dim_manager_type::DimIdxX];

                    array_index_t blockOff = pZ * (Dims > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0) +
                                             pY * (Dims > 1? gpuOffs_[dim_manager_type::DimIdxY]: 0) +
                                             pX *            gpuOffs_[dim_manager_type::DimIdxX];

                    DEBUG("TO_DEVICE: Src Block Off: %u", blockOff);

                    DEBUG("TO_DEVICE: Block (%u, %u, %u)", pZ, pY, pX);
                    DEBUG("TO_DEVICE: Dst   (%zd, %zd, %zd)", pZ * (Dims > 2? gpuOffs_[dim_manager_type::DimIdxZ]: 0),
                                                              pY * (Dims > 1? gpuOffs_[dim_manager_type::DimIdxY]: 0),
                                                              pX *            gpuOffs_[dim_manager_type::DimIdxX]);
                    DEBUG("TO_DEVICE: Src   (%zd, %zd, %zd)", pZ * localZ * (Dims > 2? this->get_dim_manager().get_offs_align()[dim_manager_type::DimIdxZ]: 0),
                                                              pY * localY * (Dims > 1? this->get_dim_manager().get_offs_align()[dim_manager_type::DimIdxY]: 0),
                                                              pX * localX);

                    myParms.dstPtr = make_cudaPitchedPtr(dataDev_ + blockOff,
                                                         sizeof(T) * hostInfo_->localDims_[dim_manager_type::DimIdxX],
                                                                     hostInfo_->localDims_[dim_manager_type::DimIdxX],
                                                         Dims > 1?   hostInfo_->localDims_[dim_manager_type::DimIdxY]: 1);

                    // We copy the whole chunk
                    myParms.dstPos = make_cudaPos(0, 0, 0);

                    myParms.srcPos = make_cudaPos(sizeof(T) * pX * localX,
                                                  pY * localY,
                                                  pZ * localZ);

                    if (PartConf::Z && pZ * localZ >= this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxZ])
                        continue;
                    if (PartConf::Y && pY * localY >= this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxY])
                        continue;
                    if (PartConf::X && pX * localX >= this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxX])
                        continue;

                    // Only transfer the remaining elements
                    if (PartConf::Z)
                    localZ = std::min(localZ, array_index_t(this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxZ] - pZ * localZ));
                    if (PartConf::Y)
                    localY = std::min(localY, array_index_t(this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxY] - pY * localY));
                    if (PartConf::X)
                    localX = std::min(localX, array_index_t(this->get_dim_manager().sizesAlign_[dim_manager_type::DimIdxX] - pX * localX));

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

    __host__ __device__ inline
    T &access_pos(array_index_t idx1, array_index_t idx2, array_index_t idx3)
    {
#ifndef __CUDA_ARCH__
        using my_linearizer = linearizer<Dims>;
        array_index_t off;
        off = my_linearizer::access_pos(this->get_dim_manager().get_offs_align(),
                                        idx1, idx2, idx3);
        return this->get_host_storage().get_addr()[off];
#else
        using my_indexer = index_block_cyclic<PartConf, BlockSize>;
        array_index_t off;
        off = my_indexer::access_pos(localOffs_,
                                     blockDims_,
                                     fakeBlockDims_,
                                     gridDims_,
                                     gpuOffs_,
                                     idx1, idx2, idx3);
        return this->dataDev_[off];
#endif
    }

    __host__ __device__ inline
    const T &access_pos(array_index_t idx1, array_index_t idx2, array_index_t idx3) const
    {
#ifndef __CUDA_ARCH__
        using my_linearizer = linearizer<Dims>;
        array_index_t off;
        off = my_linearizer::access_pos(this->get_dim_manager().get_offs_align(),
                                        idx1, idx2, idx3);
        return this->get_host_storage().get_addr()[off];
#else
        using my_indexer = index_block_cyclic<PartConf, BlockSize>;
        array_index_t off;
        off = my_indexer::access_pos(localOffs_,
                                     blockDims_,
                                     fakeBlockDims_,
                                     gridDims_,
                                     gpuOffs_,
                                     idx1, idx2, idx3);
        return this->dataDev_[off];
 #endif
    }

private:
    CUDARRAYS_TESTED(lib_storage_test, host_reshape_block_cyclic)
};

}

#endif
