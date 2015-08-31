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

#include "base.hpp"

namespace cudarrays {

template <typename T, typename StorageTraits>
class dynarray_storage<T, storage_tag::REPLICATED, StorageTraits> :
    public dynarray_base<T, StorageTraits::dimensions>
{
    static constexpr unsigned dimensions = StorageTraits::dimensions;

    using base_storage_type = dynarray_base<T, dimensions>;
    using  dim_manager_type = typename base_storage_type::dim_manager_type;

    using indexer_type = linearizer_hybrid<typename StorageTraits::offsets_seq>;

private:
    __host__
    void alloc(array_size_t elems, array_size_t offset, const std::vector<unsigned> &gpus)
    {
        DEBUG("ALLOC begin");
        for (unsigned idx : gpus) {
            unsigned gpu = (idx >= config::PEER_GPUS)? 0 : idx;

            CUDA_CALL(cudaSetDevice(gpu));
            CUDA_CALL(cudaMalloc((void **) &hostInfo_->allocsDev[idx], elems * sizeof(T)));

            DEBUG("ALLOC %u (%u) : %p", idx, gpu, hostInfo_->allocsDev[idx]);

            // Update offset
            hostInfo_->allocsDev[idx] += offset;
        }

        dataDev_ = nullptr;
    }

    T *dataDev_;

    struct storage_host_info {
        std::vector<unsigned> gpus;
        std::vector<T *> allocsDev;

        T *mergeTmp;
        T *mergeFinal;

        storage_host_info(const std::vector<unsigned> &_gpus) :
            gpus(_gpus),
            allocsDev(config::MAX_GPUS),
            mergeTmp(nullptr),
            mergeFinal(nullptr)
        {
            std::fill(allocsDev.begin(),
                      allocsDev.end(), (T *)nullptr);
        }

        ~storage_host_info()
        {
            if (mergeTmp   != nullptr) delete []mergeTmp;
            if (mergeFinal != nullptr) delete []mergeFinal;
        }
    };

    std::unique_ptr<storage_host_info> hostInfo_;

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
        if (hostInfo_) {
            // Free GPU memory
            for (unsigned gpu : utils::make_range(config::MAX_GPUS)) {
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
            for (unsigned idx : utils::make_range(mapping.comp.procs)) {
                gpus.push_back(idx);
            }

            hostInfo_.reset(new storage_host_info{gpus});

            alloc(this->get_dim_manager().get_elems_align() * sizeof(T),
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

            alloc(this->get_dim_manager().get_elems_align() * sizeof(T),
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

    T *get_dev_ptr(unsigned idx = 0)
    {
        return hostInfo_->allocsDev[idx];
    }

    const T *get_dev_ptr(unsigned idx = 0) const
    {
        return hostInfo_->allocsDev[idx];
    }

    void
    set_current_gpu(unsigned idx)
    {
        dataDev_ = hostInfo_->allocsDev[idx];
        DEBUG("Index %u > setting data dev: %p", idx, dataDev_);
    }

    unsigned get_ngpus() const
    {
        if (hostInfo_ == nullptr)
            return 0;
        return config::MAX_GPUS - std::count(hostInfo_->allocsDev.begin(),
                                             hostInfo_->allocsDev.end(), nullptr);
    }

    template <typename... Idxs>
    __device__ inline
    T &access_pos(Idxs... idxs)
    {
        auto idx = indexer_type::access_pos(this->get_dim_manager().get_strides(), idxs...);
        return dataDev_[idx];
    }

    template <typename... Idxs>
    __device__ inline
    const T &access_pos(Idxs... idxs) const
    {
        auto idx = indexer_type::access_pos(this->get_dim_manager().get_strides(), idxs...);
        return dataDev_[idx];
    }

    void to_host(host_storage &host)
    {
        ASSERT(this->get_ngpus() != 0);

        if (this->get_ngpus() == 1) {
            // Request copy-to-device
            for (unsigned idx : utils::make_range(config::MAX_GPUS)) {
                if (hostInfo_->allocsDev[idx] != nullptr) {
                    DEBUG("gpu %u > to host: %p", idx, hostInfo_->allocsDev[idx] - this->get_dim_manager().offset());
                    CUDA_CALL(cudaMemcpy(host.base_addr<T>(),
                                         hostInfo_->allocsDev[idx] - this->get_dim_manager().offset(),
                                         this->get_dim_manager().get_bytes(),
                                         cudaMemcpyDeviceToHost));
                }
            }
        } else {
            if (hostInfo_->mergeTmp == nullptr) {
                hostInfo_->mergeTmp   = new T[this->get_dim_manager().get_elems_align()];
                hostInfo_->mergeFinal = new T[this->get_dim_manager().get_elems_align()];
            }

            std::copy(host.base_addr<T>(),
                      host.base_addr<T>() + this->get_dim_manager().get_elems_align(),
                      hostInfo_->mergeFinal);

            // Request copy-to-device
            for (unsigned idx : utils::make_range(config::MAX_GPUS)) {
                if (hostInfo_->allocsDev[idx] != nullptr) {
                    CUDA_CALL(cudaMemcpy(hostInfo_->mergeTmp,
                                         hostInfo_->allocsDev[idx] - this->get_dim_manager().offset(),
                                         this->get_dim_manager().get_bytes(),
                                         cudaMemcpyDeviceToHost));

                    DEBUG("gpu %u > to host: %p", idx, hostInfo_->allocsDev[idx] - this->get_dim_manager().offset());

                    #pragma omp parallel for
                    for (array_size_t j = 0; j < this->get_dim_manager().get_elems_align(); ++j) {
                        if (memcmp(hostInfo_->mergeTmp + j, host.base_addr<T>() + j, sizeof(T)) != 0) {
                            hostInfo_->mergeFinal[j] = hostInfo_->mergeTmp[j];
                        }
                    }
                }
            }

            DEBUG("Updating main host copy");
            std::copy(hostInfo_->mergeFinal,
                      hostInfo_->mergeFinal + this->get_dim_manager().get_elems_align(),
                      host.base_addr<T>());
        }
    }

    void to_device(host_storage &host)
    {
        TRACE_FUNCTION();

        ASSERT(this->get_ngpus() != 0);

        // Request copy-to-device
        for (unsigned idx : utils::make_range(config::MAX_GPUS)) {
            if (hostInfo_->allocsDev[idx] != nullptr) {
                DEBUG("Index %u > to dev: %p", idx, hostInfo_->allocsDev[idx] - this->get_dim_manager().offset());
                CUDA_CALL(cudaMemcpy(hostInfo_->allocsDev[idx] - this->get_dim_manager().offset(),
                                     host.base_addr<T>(),
                                     this->get_dim_manager().get_bytes(),
                                     cudaMemcpyHostToDevice));
            }
        }
    }

private:
    CUDARRAYS_TESTED(lib_storage_test, host_replicated)
};

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
