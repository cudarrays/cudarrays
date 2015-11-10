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
#ifndef CUDARRAYS_DETAIL_DYNARRAY_STORAGE_REPLICATED_HPP_
#define CUDARRAYS_DETAIL_DYNARRAY_STORAGE_REPLICATED_HPP_

#include <memory>

#include "../../utils.hpp"
#include "../../system.hpp"

#include "base.hpp"

namespace cudarrays {

template <typename StorageTraits>
class dynarray_storage<detail::storage_tag::REPLICATED, StorageTraits> :
    public dynarray_base<StorageTraits>
{
    using base_storage_type = dynarray_base<StorageTraits>;
    using        value_type = typename base_storage_type::value_type;
    using    alignment_type = typename base_storage_type::alignment_type;
    using  dim_manager_type = typename base_storage_type::dim_manager_type;
    using host_storage_type = typename base_storage_type::host_storage_type;

    static constexpr auto dimensions = base_storage_type::dimensions;

    using indexer_type = linearizer_hybrid<typename StorageTraits::offsets_seq>;

private:
    __host__
    void alloc(array_size_t elems, array_size_t offset, const std::vector<unsigned> &gpus)
    {
        DEBUG("ALLOC begin");
        for (unsigned gpu : gpus) {
            CUDA_CALL(cudaSetDevice(gpu));
            CUDA_CALL(cudaMalloc((void **) &hostInfo_->allocsDev[gpu], elems * sizeof(value_type)));

            DEBUG("ALLOC in GPU %u : %p", gpu, hostInfo_->allocsDev[gpu]);

            // Update offset
            hostInfo_->allocsDev[gpu] += offset;
        }

        dataDev_ = nullptr;
    }

    value_type *dataDev_;

    struct storage_host_info {
        std::vector<unsigned> gpus;
        std::vector<value_type *> allocsDev;

        std::unique_ptr<value_type[]> mergeTmp;
        std::unique_ptr<value_type[]> mergeFinal;

        storage_host_info(const std::vector<unsigned> &_gpus) :
            gpus(_gpus),
            allocsDev(system::gpu_count())
        {
            std::fill(allocsDev.begin(),
                      allocsDev.end(), (value_type *)nullptr);
        }
    };

    std::unique_ptr<storage_host_info> hostInfo_;

public:
    __host__
    dynarray_storage(const extents<dimensions> &ext) :
        base_storage_type{ext},
        dataDev_{nullptr}
    {
    }

    __host__
    virtual ~dynarray_storage()
    {
        if (hostInfo_) {
            // Free GPU memory
            for (unsigned gpu : utils::make_range(system::gpu_count())) {
                if (hostInfo_->allocsDev[gpu] != nullptr) {
                    DEBUG("ALLOC freeing %u : %p", gpu, hostInfo_->allocsDev[gpu] - this->get_dim_manager().offset());
                    // Update offset
                    CUDA_CALL(cudaFree(hostInfo_->allocsDev[gpu] - this->get_dim_manager().offset()));
                }
            }
        }
    }

    template <size_t DimsComp>
    __host__ bool
    distribute(const cudarrays::compute_mapping<DimsComp, dimensions> &mapping)
    {
        TRACE_FUNCTION();

        bool ret = false;

        if (!hostInfo_) {
            std::vector<unsigned> gpus;
            for (unsigned gpu : utils::make_range(mapping.comp.procs)) {
                gpus.push_back(gpu);
            }

            hostInfo_.reset(new storage_host_info{gpus});

            alloc(this->get_dim_manager().get_elems_align() * sizeof(value_type),
                  this->get_dim_manager().offset(), gpus);

            ret = true;
        }

        return ret;
    }

    __host__ bool
    distribute(const std::vector<unsigned> &gpus)
    {
        TRACE_FUNCTION();

        bool ret = false;

        if (!hostInfo_) {
            hostInfo_.reset(new storage_host_info{gpus});

            alloc(this->get_dim_manager().get_elems_align() * sizeof(value_type),
                  this->get_dim_manager().offset(), gpus);

            ret = true;
        }

        return ret;
    }

    __host__ bool
    is_distributed() const
    {
        return hostInfo_ != nullptr;
    }

    value_type *get_dev_ptr(unsigned gpu = 0)
    {
        return hostInfo_->allocsDev[gpu];
    }

    const value_type *get_dev_ptr(unsigned gpu = 0) const
    {
        return hostInfo_->allocsDev[gpu];
    }

    void
    set_current_gpu(unsigned gpu)
    {
        dataDev_ = hostInfo_->allocsDev[gpu];
        DEBUG("Index %u > setting data dev: %p", gpu, dataDev_);
    }

    unsigned get_ngpus() const
    {
        if (hostInfo_ == nullptr)
            return 0;
        return system::gpu_count() - std::count(hostInfo_->allocsDev.begin(),
                                                hostInfo_->allocsDev.end(), nullptr);
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

    void to_host(host_storage_type &host)
    {
        TRACE_FUNCTION();

        ASSERT(this->get_ngpus() != 0);

        if (this->get_ngpus() == 1) {
            // Request copy-to-device. No merge required
            for (unsigned gpu : utils::make_range(system::gpu_count())) {
                if (hostInfo_->allocsDev[gpu] != nullptr) {
                    DEBUG("gpu %u > to host: %p", gpu, hostInfo_->allocsDev[gpu] - this->get_dim_manager().offset());
                    CUDA_CALL(cudaMemcpy(host.addr(),
                                         hostInfo_->allocsDev[gpu] - this->get_dim_manager().offset(),
                                         this->get_dim_manager().get_bytes(),
                                         cudaMemcpyDeviceToHost));
                }
            }
        } else {
            if (hostInfo_->mergeTmp == nullptr) {
                hostInfo_->mergeTmp.reset(new value_type[this->get_dim_manager().get_elems_align()]);
                hostInfo_->mergeFinal.reset(new value_type[this->get_dim_manager().get_elems_align()]);
            }

            std::copy(host.addr(),
                      host.addr() + this->get_dim_manager().get_elems_align(),
                      hostInfo_->mergeFinal.get());

            // Merge copies
            for (unsigned gpu : utils::make_range(system::gpu_count())) {
                if (hostInfo_->allocsDev[gpu] != nullptr) {
                    CUDA_CALL(cudaMemcpy(hostInfo_->mergeTmp.get(),
                                         hostInfo_->allocsDev[gpu] - this->get_dim_manager().offset(),
                                         this->get_dim_manager().get_bytes(),
                                         cudaMemcpyDeviceToHost));

                    DEBUG("gpu %u > to host: %p", gpu, hostInfo_->allocsDev[gpu] - this->get_dim_manager().offset());

                    // Merge step
                    #pragma omp parallel for
                    for (array_size_t j = 0; j < this->get_dim_manager().get_elems_align(); ++j) {
                        if (memcmp(hostInfo_->mergeTmp.get() + j, host.base_addr() + j, sizeof(value_type)) != 0) {
                            hostInfo_->mergeFinal[j] = hostInfo_->mergeTmp[j];
                        }
                    }
                }
            }

            DEBUG("Updating main host copy");
            std::copy(hostInfo_->mergeFinal.get(),
                      hostInfo_->mergeFinal.get() + this->get_dim_manager().get_elems_align(),
                      host.base_addr());

            // Request copy-to-device
            to_device(host);
        }
    }

    void to_device(host_storage_type &host)
    {
        TRACE_FUNCTION();

        static std::vector<cudaStream_t> streams;
        if (streams.size() == 0) {
            for (unsigned gpu : utils::make_range(system::gpu_count())) {
                CUDA_CALL(cudaSetDevice(gpu));
                cudaStream_t stream;
                CUDA_CALL(cudaStreamCreate(&stream));
                streams.push_back(stream);
            }
        }

        ASSERT(this->get_ngpus() != 0);

        // Request copy-to-device
        for (unsigned gpu : utils::make_range(system::gpu_count())) {
            if (hostInfo_->allocsDev[gpu] != nullptr) {
                DEBUG("Index %u > to dev: %p", gpu, hostInfo_->allocsDev[gpu] - this->get_dim_manager().offset());
                CUDA_CALL(cudaMemcpyAsync(hostInfo_->allocsDev[gpu] - this->get_dim_manager().offset(),
                                          host.base_addr(),
                                          this->get_dim_manager().get_bytes(),
                                          cudaMemcpyHostToDevice,
                                          streams[gpu]));
            }
        }

        // Request copy-to-device
        for (unsigned gpu : utils::make_range(system::gpu_count())) {
            CUDA_CALL(cudaStreamSynchronize(streams[gpu]));
        }
    }

private:
    CUDARRAYS_TESTED(lib_storage_test, host_replicated)
};

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
