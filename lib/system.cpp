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

} // namespace system
} // namespace cudarrays
