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

#include "base.hpp"

namespace cudarrays {

template <typename T, unsigned Dims, typename PartConf>
class dynarray_storage<T, Dims, REPLICATED, PartConf> :
    public dynarray_base<T, Dims>
{
    using base_storage_type = dynarray_base<T, Dims>;
    using  dim_manager_type = typename base_storage_type::dim_manager_type;
    using      extents_type = typename base_storage_type::extents_type;

    using indexer_type = linearizer<Dims>;

private:
    __host__
    void alloc(array_size_t elems, array_size_t offset, const std::vector<unsigned> &gpus)
    {
        DEBUG("Replicated> ALLOC begin");
        for (unsigned idx : gpus) {
            unsigned gpu = (idx >= config::PEER_GPUS)? 0 : idx;

            CUDA_CALL(cudaSetDevice(gpu));
            CUDA_CALL(cudaMalloc((void **) &hostInfo_->allocsDev[idx], elems * sizeof(T)));

            DEBUG("Replicated> ALLOC %u (%u) : %p", idx, gpu, hostInfo_->allocsDev[idx]);

            // Update offset
            hostInfo_->allocsDev[idx] += offset;
        }

        dataDev_ = nullptr;
    }

    T *dataDev_;

    struct storage_host_info {
        unsigned gpus = 0;
        std::vector<T *> allocsDev;

        storage_host_info(unsigned _gpus) :
            gpus(_gpus),
            allocsDev(config::MAX_GPUS)
        {
            std::fill(allocsDev.begin(),
                      allocsDev.end(), (T *)nullptr);
        }
    };

    storage_host_info *hostInfo_;

public:
    __host__
    dynarray_storage(const extents_type &extents,
                  const align_t &align) :
        base_storage_type(extents, align),
        dataDev_(nullptr),
        hostInfo_(nullptr)
    {
    }

    __host__
    virtual ~dynarray_storage()
    {
        if (hostInfo_ != nullptr) {
            // Free GPU memory
            for (unsigned gpu : utils::make_range(config::MAX_GPUS)) {
                if (hostInfo_->allocsDev[gpu] != nullptr) {
                    DEBUG("Replicated> ALLOC freeing %u : %p", gpu, hostInfo_->allocsDev[gpu] - this->get_dim_manager().get_offset());
                    // Update offset
                    CUDA_CALL(cudaFree(hostInfo_->allocsDev[gpu] - this->get_dim_manager().get_offset()));
                }
            }

            delete hostInfo_;
        }
    }

    template <size_t DimsComp>
    __host__ bool
    distribute(const compute_mapping<DimsComp, Dims> &mapping)
    {
        DEBUG("Replicated> Distributing");
        if (!hostInfo_) {
            hostInfo_ = new storage_host_info{mapping.comp.procs};

            std::vector<unsigned> gpus;
            for (unsigned idx : utils::make_range(hostInfo_->gpus)) {
                gpus.push_back(idx);
            }
            alloc(this->get_dim_manager().get_elems_align(), this->get_dim_manager().get_offset(), gpus);

            return true;
        }
        return false;
    }

    __host__ bool
    distribute(const std::vector<unsigned> &gpus)
    {
        DEBUG("Replicated> Distributing2");
        if (!hostInfo_) {
            hostInfo_ = new storage_host_info{gpus.size()};

            alloc(this->get_dim_manager().get_elems_align(), this->get_dim_manager().get_offset(), gpus);

            return true;
        }
        return false;
    }

    __host__ bool
    is_distributed()
    {
        return hostInfo_->allocsDev != nullptr;
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
        DEBUG("Replicated> Index %u > setting data dev: %p", idx, dataDev_);
    }

    unsigned get_ngpus() const
    {
        if (hostInfo_ == nullptr)
            return 0;
        return std::count_if(hostInfo_->allocsDev.begin(),
                             hostInfo_->allocsDev.end(),
                             [](T *ptr) { return ptr != nullptr; });
    }

    __array_index__
    T &access_pos(array_index_t idx1, array_index_t idx2, array_index_t idx3)
    {
        auto idx = indexer_type::access_pos(this->get_dim_manager().get_offs_align(), idx1, idx2, idx3);
        return
#ifndef __CUDA_ARCH__
               this->get_host_storage().get_addr()
#else
               dataDev_
#endif
               [idx];

    }

    __array_index__
    const T &access_pos(array_index_t idx1, array_index_t idx2, array_index_t idx3) const
    {
        auto idx = indexer_type::access_pos(this->get_dim_manager().get_offs_align(), idx1, idx2, idx3);
        return
#ifndef __CUDA_ARCH__
               this->get_host_storage().get_addr()
#else
               dataDev_
#endif
               [idx];

    }

    void to_host()
    {
        if (this->get_ngpus()) {
            // Request copy-to-device
            for (unsigned idx : utils::make_range(config::MAX_GPUS)) {
                if (hostInfo_->allocsDev[idx] != nullptr) {
                    DEBUG("Replicated> gpu %u > to host: %p", idx, hostInfo_->allocsDev[idx] - this->get_dim_manager().get_offset());
                    CUDA_CALL(cudaMemcpy(this->get_host_storage().get_base_addr(),
                                         hostInfo_->allocsDev[idx] - this->get_dim_manager().get_offset(),
                                         this->get_dim_manager().get_elems_align() * sizeof(T),
                                         cudaMemcpyDeviceToHost));
                }
            }
        } else {
            std::unique_ptr<T[]> tmp{new T[this->get_dim_manager().get_elems_align()]};
            std::unique_ptr<T[]> merged{new T[this->get_dim_manager().get_elems_align()]};

            std::copy(this->get_host_storage().get_base_addr(),
                      this->get_host_storage().get_base_addr() + this->get_dim_manager().get_elems_align(),
                      merged.get());

            // Request copy-to-device
            for (unsigned idx : utils::make_range(config::MAX_GPUS)) {
                if (hostInfo_->allocsDev[idx] != nullptr) {
                    CUDA_CALL(cudaMemcpy(tmp.get(),
                                         hostInfo_->allocsDev[idx] - this->get_dim_manager().get_offset(),
                                         this->get_dim_manager().get_elems_align() * sizeof(T),
                                         cudaMemcpyDeviceToHost));

                    DEBUG("Replicated> gpu %u > to host: %p", idx, hostInfo_->allocsDev[idx] - this->get_dim_manager().get_offset());

                    #pragma omp parallel for
                    for (array_size_t j = 0; j < this->get_dim_manager().get_elems_align(); ++j) {
                        if (memcmp(tmp.get() + j, this->get_host_storage().get_base_addr() + j, sizeof(T)) != 0) {
                            merged[j] = tmp[j];
                        }
                    }
                }
            }

            DEBUG("Replicated> Updating main host copy");
            std::copy(merged.get(),
                      merged.get() + this->get_dim_manager().get_elems_align(),
                      this->get_host_storage().get_base_addr());
        }
    }

    void to_device()
    {
        // Request copy-to-device
        for (unsigned idx : utils::make_range(config::MAX_GPUS)) {
            if (hostInfo_->allocsDev[idx] != nullptr) {
                DEBUG("Replicated> Index %u > to dev: %p", idx, hostInfo_->allocsDev[idx] - this->get_dim_manager().get_offset());
                CUDA_CALL(cudaMemcpy(hostInfo_->allocsDev[idx] - this->get_dim_manager().get_offset(),
                                     this->get_host_storage().get_base_addr(),
                                     this->get_dim_manager().get_elems_align() * sizeof(T),
                                     cudaMemcpyHostToDevice));
            }
        }
    }

private:
    CUDARRAYS_TESTED(lib_storage_test, host_replicated)
};

}

#endif
