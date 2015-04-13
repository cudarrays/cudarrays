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

#include <algorithm>
#include <cstdlib>
#include <cstddef>

#include <cuda_runtime.h>

#include "cudarrays/common.hpp"
#include "cudarrays/memory.hpp"
#include "cudarrays/utils.hpp"

#include "cudarrays/detail/utils/log.hpp"

namespace cudarrays {

// __attribute__((constructor(65535)))
void init_lib()
{
    // Initialize logging
    config::OPTION_DEBUG = utils::getenv<bool>("CUDARRAYS_DEBUG", false);

    // Get gpus from environment variable
    config::MAX_GPUS = utils::getenv<unsigned>("CUDARRAYS_GPUS", 0);

    // VM alignment used by the CUDA driver
    config::CUDA_VM_ALIGN = utils::getenv<array_size_t>("CUDARRAYS_VM_ALIGN", 1 * 1024 * 1024);

    // Page size to be emulated in VM based allocators
    config::PAGE_ALIGN = utils::getenv<array_size_t>("CUDARRAYS_PAGE_ALIGN", 4 * 1024);

    config::PAGES_PER_ARENA = config::CUDA_VM_ALIGN / config::PAGE_ALIGN;

    DEBUG("Inizializing CUDArrays");
    if (config::MAX_GPUS != 0)
        DEBUG("- Max GPUS: %zd", config::MAX_GPUS);
    else
        DEBUG("- Max GPUS: autodetect");

    cudaError_t err;

    int devices;
    err = cudaGetDeviceCount(&devices);
    assert(err == cudaSuccess);

    for (int d1 = 0; d1 < devices; ++d1) {
        err = cudaSetDevice(d1);
        assert(err == cudaSuccess);
        unsigned peers = 1;
        for (int d2 = 0; d2 < devices; ++d2) {
            if (d1 != d2) {
                int access;
                err = cudaDeviceCanAccessPeer(&access, d1, d2);
                assert(err == cudaSuccess);

                if (access) {
                    err = cudaDeviceEnablePeerAccess(d2, 0);
                    assert(err == cudaSuccess);

                    ++peers;
                }
            }
        }
#if CUDARRAYS_DEBUG_CUDA == 1
        err = cudaSetDevice(d1);
        assert(err == cudaSuccess);
        size_t value;
        err = cudaDeviceGetLimit(&value, cudaLimitStackSize);
        assert(err == cudaSuccess);
        err = cudaDeviceSetLimit(cudaLimitStackSize, value * 4);
        assert(err == cudaSuccess);

        printf("GPU %u: Increasing stack size to %zd\n", d1, value * 2);
#endif

        config::PEER_GPUS = std::max(config::PEER_GPUS, peers);
    }

    // WARM-UP
    if (0)
    {
        size_t S = 512 * 1024;
        std::vector<char *> ptrs;

        for (int d = 0; d < devices; ++d) {
            char *curr;

            err = cudaSetDevice(d);
            assert(err == cudaSuccess);
            err = cudaMalloc((void **) &curr, S);
            assert(err == cudaSuccess);
            ptrs.push_back(curr);
        }

        for (auto ptr : ptrs) {
            err = cudaFree(ptr);
            assert(err == cudaSuccess);
        }
    }

    if (config::MAX_GPUS == 0)
        config::MAX_GPUS = config::PEER_GPUS;

    handler_sigsegv_overload();

    DEBUG("- Peer GPUS: %zd", config::PEER_GPUS);

    // Merge information provided by the compiler
    cudarrays_compiler_register_info__();
}

//__attribute__((destructor(65535)))
void fini_lib()
{
    handler_sigsegv_restore();
}

}
