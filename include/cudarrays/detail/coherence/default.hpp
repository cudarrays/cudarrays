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

class default_coherence :
    public coherence_policy {
public:
    enum location : unsigned {
        GPU    = 0b01,
        CPU    = 0b10,
        SHARED = 0b11
    };

    enum class ownership {
        GPU,
        CPU
    };

    default_coherence() :
        obj_(nullptr),
        location_(location::CPU),
        owner_(ownership::CPU)
    {
    }

    virtual ~default_coherence()
    {
        unbind();
    }

    void bind(coherent &obj)
    {
        if (!obj_) {
            obj_ = &obj;

            register_range(obj_->host_addr(),
                           obj_->size());
        }
    }

    void unbind()
    {
        if (obj_) {
            unregister_range(obj_->host_addr());
            obj_ = nullptr;
        }
    }

    bool is_bound() const
    {
        return obj_ != nullptr;
    }

    void release(const std::vector<unsigned> &gpus, bool Const)
    {
        DEBUG("Coherence> Release: %p", obj_->host_addr());

        if (owner_ != ownership::GPU) {
            DEBUG("Coherence> Ownership -> GPU");
            owner_ = ownership::GPU;
        }

        if (!obj_->is_distributed()) {
            bool ok = obj_->distribute(gpus);
            ASSERT(ok, "Error while distributing array");
        }

        if (location_ == location::CPU) {
            DEBUG("Coherence> obj TO DEVICE: %p", obj_->host_addr());
            obj_->to_device();

            if (!Const) {
                location_ = location::GPU;
                DEBUG("Coherence> CPU -> GPU");
            } else {
                location_ = location::SHARED;
                DEBUG("Coherence> CPU -> SHARED");
            }
        } else if (location_ == location::GPU) {
        } else { // location_ == SHARED
            if (!Const) {
                location_ = location::GPU;
                DEBUG("Coherence> SHARED -> GPU");
            }
        }

        // Protect memory so that it is not accessible during GPU execution
        protect_range(obj_->host_addr(),
                      obj_->size(),
                      [this](bool write) -> bool
                      {
                          void *ptr = obj_->host_addr();

                          unprotect_range(ptr);

                          DEBUG("Coherence> obj TO HOST: %p", ptr);
                          obj_->to_host();

                          if (write) {
                            location_ = location::CPU;
                            DEBUG("Coherence> GPU -> CPU");
                          } else {
                            location_ = location::SHARED;
                            DEBUG("Coherence> GPU -> SHARED");
                          }

                          return true;
                      });
    }

    void acquire()
    {
        DEBUG("Coherence> Acquire: %p", obj_->host_addr());

        ASSERT(owner_ == ownership::GPU);

        DEBUG("Coherence> Ownership -> CPU");
        owner_ = ownership::CPU;
    }

private:
    coherent *obj_;
    location location_;
    ownership owner_;
};

}

#endif

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
