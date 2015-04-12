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
#ifndef CUDARRAYS_DIST_COHERENCE_HPP_
#define CUDARRAYS_DIST_COHERENCE_HPP_

#include "common.hpp"
#include "memory.hpp"

#include "detail/utils/log.hpp"

namespace cudarrays {

class coherence_policy {
public:
    virtual void release(std::vector<unsigned> gpus, bool Const) = 0;
    virtual void acquire() = 0;
};

class coherent {
public:
    virtual coherence_policy &get_coherence_policy() = 0;
    virtual void set_current_gpu(unsigned /*idx*/) = 0;
};

template <typename ArrayType>
class default_coherence :
    public coherence_policy {
public:
    enum state {
        SHARED = 1,
        GPU    = 2,
        CPU    = 4
    };

    default_coherence() :
        array_(nullptr),
        state_(CPU)
    {
    }

    void bind(ArrayType *array)
    {
        // TODO: improve binding logic
        if (!array_)
            array_ = array;
    }

    bool is_bound() const
    {
        return array_ != nullptr;
    }

    void release(std::vector<unsigned> /*gpus*/, bool Const)
    {
#if 0
        if (!arg.is_distributed()) {
            comp_conf<1> confComp = { comp1D_none, 1 };

            comp_mapping<1, Dims> mapping;
            mapping.comp = confComp;

            arg.template distribute<1>(mapping);
        }
#endif
        DEBUG("Coherence> Release: %p", array_->get_host_storage().get_base_addr());

        if (state_ == CPU) {
            DEBUG("Coherence> Array TO DEVICE: %p", array_->get_host_storage().get_base_addr());
            array_->to_device();

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
        DEBUG("Coherence> Acquire: %p", array_->get_host_storage().get_base_addr());

        if (state_ == GPU) {
            // Delay acquire
            protect_range(array_->get_host_storage().get_base_addr(),
                          array_->get_host_storage().size(),
                          [this](bool write) -> bool
                          {
                              void *ptr = array_->get_host_storage().get_base_addr();

                              unprotect_range(ptr);

                              DEBUG("Coherence> Array TO HOST: %p", ptr);
                              array_->to_host();

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
    ArrayType *array_;
    state state_;
};

}

#endif
