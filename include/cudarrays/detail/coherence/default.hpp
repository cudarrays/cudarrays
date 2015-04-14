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
#ifndef CUDARRAYS_DETAIL_COHERENCE_DEFAULT_HPP_
#define CUDARRAYS_DETAIL_COHERENCE_DEFAULT_HPP_

#include "../../coherence.hpp"

namespace cudarrays {

// TODO: Remove template, only accept objects that inherit from coherent
template <typename Coherent>
class default_coherence :
    public coherence_policy {
public:
    enum state {
        SHARED = 1,
        GPU    = 2,
        CPU    = 4
    };

    default_coherence() :
        obj_(nullptr),
        state_(CPU)
    {
    }

    void bind(Coherent *obj)
    {
        // TODO: improve binding logic
        if (!obj_)
            obj_ = obj;
    }

    bool is_bound() const
    {
        return obj_ != nullptr;
    }

    void release(std::vector<unsigned> gpus, bool Const)
    {
        if (!obj_->is_distributed()) {
            bool ok = obj_->distribute(gpus);
            ASSERT(ok, "Error while distributing array");
        }

        DEBUG("Coherence> Release: %p", obj_->get_host_storage().get_base_addr());

        if (state_ == CPU) {
            DEBUG("Coherence> obj TO DEVICE: %p", obj_->get_host_storage().get_base_addr());
            obj_->to_device();

            if (!Const) {
                state_ = GPU;
                DEBUG("Coherence> CPU -> GPU");
            } else {
                state_ = SHARED;
                DEBUG("Coherence> CPU -> SHARED");
            }
        } else if (state_ == GPU) {
        } else { // state_ == SHARED
            if (!Const) {
                state_ = GPU;
                DEBUG("Coherence> SHARED -> GPU");
            }
        }
    }

    void acquire()
    {
        DEBUG("Coherence> Acquire: %p", obj_->get_host_storage().get_base_addr());

        if (state_ == GPU) {
            // Delay acquire
            protect_range(obj_->get_host_storage().get_base_addr(),
                          obj_->get_host_storage().size(),
                          [this](bool write) -> bool
                          {
                              void *ptr = obj_->get_host_storage().get_base_addr();

                              unprotect_range(ptr);

                              DEBUG("Coherence> obj TO HOST: %p", ptr);
                              obj_->to_host();

                              if (write) {
                                state_ = CPU;
                                DEBUG("Coherence> GPU -> CPU");
                              } else {
                                state_ = SHARED;
                                DEBUG("Coherence> GPU -> SHARED");
                              }

                              return true;
                          }, true);
        }
    }

private:
    Coherent *obj_;
    state state_;
};

}

#endif
