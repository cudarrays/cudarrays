#include <cuda_runtime_api.h>

#include <vector>

#include "cudarrays/system.hpp"

namespace cudarrays {

namespace system {

// Get GPUs from environment variable
utils::option<unsigned> MAX_GPUS{"CUDARRAYS_MAX_GPUS", 0};
utils::option<array_size_t> CUDA_VM_ALIGN{"CUDARRAYS_VM_ALIGN", 1 * 1024 * 1024};

unsigned GPUS;
unsigned PEER_GPUS;

unsigned
gpu_count()
{
    // LIBRARY ENTRY POINT
    cudarrays_entry_point();

    return GPUS;
}

unsigned
peer_gpu_count()
{
    // LIBRARY ENTRY POINT
    cudarrays_entry_point();

    return PEER_GPUS;
}

void init()
{
    cudaError_t err;

    int devices;
    err = cudaGetDeviceCount(&devices);
    ASSERT(err == cudaSuccess);

    if (MAX_GPUS.value() == 0)
        GPUS = devices;

    for (int d1 = 0; d1 < devices; ++d1) {
        err = cudaSetDevice(d1);
        ASSERT(err == cudaSuccess);
        unsigned peers = 1;
        for (int d2 = 0; d2 < devices; ++d2) {
            if (d1 != d2) {
                int access;
                err = cudaDeviceCanAccessPeer(&access, d1, d2);
                ASSERT(err == cudaSuccess);

                if (access) {
                    err = cudaDeviceEnablePeerAccess(d2, 0);
                    ASSERT(err == cudaSuccess);

                    ++peers;
                }
            }
        }
#if CUDARRAYS_DEBUG_CUDA == 1
        err = cudaSetDevice(d1);
        ASSERT(err == cudaSuccess);
        size_t value;
        err = cudaDeviceGetLimit(&value, cudaLimitStackSize);
        ASSERT(err == cudaSuccess);
        err = cudaDeviceSetLimit(cudaLimitStackSize, value * 4);
        ASSERT(err == cudaSuccess);

        printf("GPU %u: Increasing stack size to %zd\n", d1, value * 2);
#endif
        PEER_GPUS = std::max(PEER_GPUS, peers);
    }

    for (unsigned i = 0; i < GPUS; ++i) {
        err = cudaSetDevice(i);
        ASSERT(err == cudaSuccess);

        // Preallocate streams for kernel execution
        cudaStream_t stream;
        err = cudaStreamCreate(&stream);
        ASSERT(err == cudaSuccess);
        cudarrays::StreamsIn.push_back(stream);
        err = cudaStreamCreate(&stream);
        ASSERT(err == cudaSuccess);
        cudarrays::StreamsOut.push_back(stream);
    }
}

} // namespace system
} // namespace cudarrays
