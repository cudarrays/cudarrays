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
#ifndef CUDARRAYS_DETAIL_DYNARRAY_STORAGE_VM_HPP_
#define CUDARRAYS_DETAIL_DYNARRAY_STORAGE_VM_HPP_

#include <memory>

#include "../../system.hpp"
#include "../../utils.hpp"

#include "base.hpp"
#include "helpers.hpp"

namespace cudarrays {

template <unsigned Dims>
class page_allocator {
public:
    struct page_stats {
        unsigned gpu = 0;

        array_size_t local;
        array_size_t remote;

        page_stats() :
            gpu{0},
            local{0},
            remote{0}
        {}

        array_size_t get_total() const
        {
            return local + remote;
        }

        double get_imbalance_ratio() const
        {
            return double(remote) / double(get_total());
        }
    };

    page_allocator(unsigned gpus,
                   const std::array<array_size_t, Dims> &elems,
                   const std::array<array_size_t, Dims> &elems_align,
                   const std::array<array_size_t, Dims> &elems_local,
                   const std::array<unsigned, Dims> &arrayDimToGpus,
                   array_size_t granularity) :
        gpus_{gpus},
        dims_(elems),
        dimsAlign_(elems_align),
        dimsLocal_(elems_local),
        arrayDimToGpus_(arrayDimToGpus),
        granularity_{granularity}
    {
        utils::fill(idx_, array_index_t(0));

        nelems_ = utils::accumulate(elems, 1, std::multiplies<array_size_t>());

        for (unsigned dim = 0; dim < Dims; ++dim) {
            mayOverflow_[dim] = elems_align[dim] % elems_local[dim] != 0;
        }
    }


    std::pair<bool, page_stats>
    advance()
    {
        bool done = false;
        array_size_t inc = granularity_;

        std::vector<array_size_t> localStats(gpus_, 0); // gpu -> local_elems

        if (idx_[0] == dimsAlign_[0]) return std::make_pair(true, page_stats{});

        do {
            std::array<unsigned, Dims> tileId;

            for (auto dim : utils::make_range(Dims)) {
                tileId[dim] = idx_[dim] / dimsLocal_[dim];
            }
            unsigned gpu = 0;
            for (auto dim : utils::make_range(Dims)) {
                gpu += tileId[dim] * arrayDimToGpus_[dim];
            }

            // Compute increment until the next tile in the lowest-order dimension
            auto off = dimsLocal_[Dims - 1] - (idx_[Dims - 1] % dimsLocal_[Dims - 1]);
            // Check if array dimensions are not divisible by local tile dimensions
            if (//mayOverflow_[Dims - 1] &&
                idx_[Dims - 1] + off >= dimsAlign_[Dims - 1])
                off = dimsAlign_[Dims - 1] - idx_[Dims - 1];

            if (inc < off) { // We are done
                idx_[Dims - 1] += inc;

                localStats[gpu] += inc;

                inc = 0;
            } else {
                if (idx_[Dims - 1] + off == dimsAlign_[Dims - 1]) {
                    if (Dims > 1) {
                        idx_[Dims - 1] = 0;
                        // End of a dimension propagate the carry
                        for (int dim = int(Dims) - 2; dim >= 0; --dim) {
                            ++idx_[dim];

                            if (idx_[dim] < dimsAlign_[dim]) {
                                break; // We don't have to propagate the carry further
                            } else if (dim > 0 && idx_[dim] == dimsAlign_[dim]) {
                                // Set index to 0 unless for the highest-order dimension
                                idx_[dim] = 0;
                            }
                        }
                        if (idx_[0] == dimsAlign_[0]) {
                            done = true;
                            localStats[gpu] += off;
                            break;
                        }
                    } else {
                        done = true;
                        idx_[Dims - 1] += off;
                    }
                } else {
                    // Next tile within the dimension
                    idx_[Dims - 1] += off;
                }
                inc -= off;

                localStats[gpu] += off;
            }
        } while (idx_[0] != dimsAlign_[0] && inc > 0);

        // Implement some way to balance allocations on 50% imbalance
        unsigned mgpu = 0;
        array_size_t sum = 0;
        for (auto gpu : utils::make_range(gpus_)) {
            if (localStats[gpu] > localStats[mgpu]) mgpu = gpu;

            sum += localStats[gpu];
        }

        page_stats page;
        page.gpu    = mgpu;
        page.local  = localStats[mgpu];
        page.remote = sum - localStats[mgpu];

        pageStats_.push_back(page);

        return std::pair<bool, page_stats>(done, page);
    }

    double get_imbalance_ratio() const
    {
        array_size_t local  = 0;
        array_size_t remote = 0;

        for (const page_stats &page : pageStats_) {
            local  += page.local;
            remote += page.remote;
        }
        if (local + remote == 0)
            return 0.0;

        return double(remote)/double(local + remote);
    }

private:
    unsigned gpus_;

    extents<Dims> dims_;
    extents<Dims> dimsAlign_;
    extents<Dims> dimsLocal_;

    std::array<unsigned, Dims> arrayDimToGpus_;

    std::array<array_size_t, Dims> idx_;

    array_size_t nelems_;
    array_size_t granularity_;

    std::array<bool, Dims> mayOverflow_;

    std::vector<page_stats> pageStats_;

    CUDARRAYS_TESTED(storage_test, vm_page_allocator1)
    CUDARRAYS_TESTED(storage_test, vm_page_allocator2)
};

template <typename StorageTraits>
class dynarray_storage<detail::storage_tag::VM, StorageTraits> :
    public dynarray_base<StorageTraits>
{
    using base_storage_type = dynarray_base<StorageTraits>;
    using        value_type = typename base_storage_type::value_type;
    using    alignment_type = typename base_storage_type::alignment_type;
    using  dim_manager_type = typename base_storage_type::dim_manager_type;
    using host_storage_type = typename base_storage_type::host_storage_type;

    static constexpr auto dimensions = base_storage_type::dimensions;

    using indexer_type = linearizer_hybrid<typename StorageTraits::offsets_seq>;

    struct storage_host_info {
        unsigned gpus;

        extents<dimensions> localDims;
        std::array<unsigned, dimensions> arrayDimToGpus;

        unsigned npages;

        storage_host_info(unsigned _gpus) :
            gpus{_gpus},
            npages{0}
        {
        }
    };

public:
    template <unsigned DimsComp>
    __host__ bool
    distribute(const cudarrays::compute_mapping<DimsComp, dimensions> &mapping)
    {
        if (!dataDev_) {
            hostInfo_.reset(new storage_host_info{mapping.comp.procs});

            if (mapping.comp.procs == 1) {
                hostInfo_->localDims = this->get_dim_manager().dims();
                utils::fill(hostInfo_->arrayDimToGpus, 0);
                alloc(mapping.comp.procs);
            } else {
                compute_distribution_internal(mapping);
                // TODO: remove when the allocation is properly done
                alloc(mapping.comp.procs);
            }

            return true;
        }
        return false;
    }

    __host__ bool
    distribute(const std::vector<unsigned> &)
    {
#if 0
        if (!dataDev_) {
            hostInfo_.reset(new storage_host_info(mapping.comp.procs));

            if (mapping.comp.procs == 1) {
                hostInfo_->localDims = make_array(this->get_dim_manager().dims());
                fill(hostInfo_->arrayDimToGpus, 0);
                alloc(mapping.comp.procs);
            } else {
                compute_distribution(mapping);
                // TODO: remove when the allocation is properly done
                alloc(mapping.comp.procs);

                //DEBUG("ALLOC: SHIIIIT: %s", to_string(offsGPU_, Dims - 1));
            }

            return true;
        }
#endif
        return false;
    }

    __host__ bool
    is_distributed() const
    {
        return dataDev_ != NULL;
    }

    unsigned get_ngpus() const
    {
        return hostInfo_->gpus;
    }

    __host__
    void to_host(host_storage_type &host)
    {
        unsigned npages;
        npages = utils::div_ceil(host.size(), system::CUDA_VM_ALIGN.value());

        value_type *src = dataDev_ - this->get_dim_manager().offset();
        value_type *dst = host.base_addr();
        for (array_size_t idx  = 0; idx < npages; ++idx) {
            array_size_t bytesChunk = system::CUDA_VM_ALIGN;
            if ((idx + 1) * system::CUDA_VM_ALIGN > host.size())
                bytesChunk = host.size() - idx * system::CUDA_VM_ALIGN;

            DEBUG("COPYING TO HOST: %p -> %p (%zd)", &src[(system::CUDA_VM_ALIGN * idx)/sizeof(value_type)],
                                                     &dst[(system::CUDA_VM_ALIGN * idx)/sizeof(value_type)],
                                                     size_t(bytesChunk));
            CUDA_CALL(cudaMemcpy(&dst[(system::CUDA_VM_ALIGN * idx)/sizeof(value_type)],
                                 &src[(system::CUDA_VM_ALIGN * idx)/sizeof(value_type)],
                                 bytesChunk, cudaMemcpyDeviceToHost));
        }
    }

    __host__
    void to_device(host_storage_type &host)
    {
        unsigned npages;
        npages = utils::div_ceil(host.size(), system::CUDA_VM_ALIGN.value());

        value_type *src = host.base_addr();
        value_type *dst = dataDev_ - this->get_dim_manager().offset();
        for (array_size_t idx  = 0; idx < npages; ++idx) {
            array_size_t bytesChunk = system::CUDA_VM_ALIGN;
            if ((idx + 1) * system::CUDA_VM_ALIGN > host.size())
                bytesChunk = host.size() - idx * system::CUDA_VM_ALIGN;

            DEBUG("COPYING TO DEVICE: %p -> %p (%zd)", &src[(system::CUDA_VM_ALIGN * idx)/sizeof(value_type)],
                                                       &dst[(system::CUDA_VM_ALIGN * idx)/sizeof(value_type)],
                                                       size_t(bytesChunk));
            CUDA_CALL(cudaMemcpy(&dst[(system::CUDA_VM_ALIGN * idx)/sizeof(value_type)],
                                 &src[(system::CUDA_VM_ALIGN * idx)/sizeof(value_type)],
                                 bytesChunk, cudaMemcpyHostToDevice));
        }
    }

    __host__
    dynarray_storage(const extents<dimensions> &ext) :
        base_storage_type{ext},
        dataDev_{nullptr},
        hostInfo_{nullptr}
    {
    }

    __host__
    dynarray_storage(const dynarray_storage &other) :
        base_storage_type(other),
        dataDev_{other.dataDev_},
        hostInfo_{other.hostInfo_}
    {
    }

    __host__
    virtual ~dynarray_storage()
    {
#ifndef __CUDA_ARCH__
        if (dataDev_ != NULL) {
            // Get base address
            value_type *data = dataDev_ - this->get_dim_manager().offset();

            // Free each page in GPU memory
            for (auto idx : utils::make_range(hostInfo_->npages)) {
                CUDA_CALL(cudaFree(&data[system::vm_cuda_align_elems<value_type>() * idx]));
            }
        }
#endif
    }

    template <typename... Idxs>
    __device__ inline
    value_type &access_pos(Idxs&&... idxs)
    {
        auto idx = indexer_type::access_pos(this->get_dim_manager().get_strides(), std::forward<Idxs>(idxs)...);
        return dataDev_[idx];
    }

    template <typename... Idxs>
    __device__ inline
    const value_type &access_pos(Idxs&&... idxs) const
    {
        auto idx = indexer_type::access_pos(this->get_dim_manager().get_strides(), std::forward<Idxs>(idxs)...);
        return dataDev_[idx];
    }


private:
    __host__
    void alloc(unsigned gpus)
    {
        TRACE_FUNCTION();

        using my_allocator = page_allocator<dimensions>;

        extents<dimensions> elems;
        extents<dimensions> elemsAlign;

        utils::copy(this->get_dim_manager().dims(), elems);
        utils::copy(this->get_dim_manager().dims_align(), elemsAlign);

        my_allocator cursor(gpus,
                            elems,
                            elemsAlign,
                            hostInfo_->localDims,
                            hostInfo_->arrayDimToGpus,
                            system::vm_cuda_align_elems<value_type>());

        char *last = NULL, *curr = NULL;

        unsigned npages = 0;

        // Allocate data in the GPU memory
        for (auto ret = cursor.advance();
             !ret.first || ret.second.get_total() > 0;
             ret = cursor.advance()) {
            typename my_allocator::page_stats page = ret.second;

            CUDA_CALL(cudaSetDevice(page.gpu));

            CUDA_CALL(cudaMalloc((void **) &curr, system::CUDA_VM_ALIGN));

            DEBUG("ALLOCATING: %zd bytes in %u (%zd)",
                  system::CUDA_VM_ALIGN.value(), page.gpu, size_t(curr) % system::CUDA_VM_ALIGN);

            DEBUG("ALLOCATE: %p in %u (final: %u total: %zd)",
                  curr, page.gpu, unsigned(ret.first), size_t(ret.second.get_total()));

            if (last == NULL) {
                dataDev_ = (value_type *) curr;
            } else {
                // Check if the driver is allocating contiguous virtual addresses
                ASSERT(last + system::CUDA_VM_ALIGN == curr);
            }
            std::swap(last, curr);
            ++npages;
        }

        hostInfo_->npages = npages;

        dataDev_ += this->get_dim_manager().offset();
    }

    template <unsigned DimsComp>
    __host__
    void compute_distribution_internal(const cudarrays::compute_mapping<DimsComp, dimensions> &mapping)
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
        //hostInfo_->arrayPartitionGrid = arrayPartitionGrid;
        // 3- Compute dimensions of each tile
        std::array<array_size_t, dimensions> localDims = helper_distribution_get_local_dims(dims, arrayPartitionGrid);
        hostInfo_->localDims = localDims;
        // 4- Compute the GPU grid offsets (iterate from lowest-order dimension)
        std::array<unsigned, DimsComp> gpuGridOffs = helper_distribution_gpu_get_offs(gpuGrid);
        // 5- Compute the array to GPU mapping needed for the allocation based on the grid offsets
        std::array<unsigned, dimensions> arrayDimToGpus = helper_distribution_get_array_dim_to_gpus(gpuGridOffs, arrayDimToCompDim);
        hostInfo_->arrayDimToGpus = arrayDimToGpus;

        DEBUG("BASE INFO");
        DEBUG("- array dims: %s", dims);
        DEBUG("- comp  dims: %u", DimsComp);
        DEBUG("- comp -> array: %s", arrayDimToCompDim);

        DEBUG("PARTITIONING");
        DEBUG("- gpus: %u", mapping.comp.procs);
        DEBUG("- comp  part: %s (%u)", mapping.comp.info, mapping.comp.get_part_dims());
        DEBUG("- comp  grid: %s", gpuGrid);
        DEBUG("- array grid: %s", arrayPartitionGrid);
        DEBUG("- local dims: %s", hostInfo_->localDims);

        DEBUG("- array grid offsets: %s", arrayDimToGpus);
    }

    value_type *dataDev_;

    std::unique_ptr<storage_host_info> hostInfo_;

    CUDARRAYS_TESTED(lib_storage_test, host_vm)
};

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
