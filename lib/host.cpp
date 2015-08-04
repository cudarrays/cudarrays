#include <sys/mman.h>

#include "cudarrays/utils.hpp"
#include "cudarrays/host.hpp"

namespace cudarrays {

void
host_storage::free_data()
{
    int ret = munmap(this->base_addr<void>(), state_->hostSize_);
    ASSERT(ret == 0);
    state_->data_ = nullptr;
}


host_storage::host_storage()
{
    state_ = new state;
    state_->data_ = nullptr;
}

host_storage::~host_storage()
{
    if (state_->data_ != nullptr) {
        free_data();
    }
    delete state_;
    state_ = nullptr;
}

void
host_storage::alloc(array_size_t bytes, array_size_t offset, void *addr)
{
    int flags = MAP_PRIVATE | MAP_ANONYMOUS;
    if (addr != nullptr) flags |= MAP_FIXED;
    state_->hostSize_ = bytes;
    state_->data_ = mmap(addr, state_->hostSize_,
            PROT_READ | PROT_WRITE,
            flags, -1, 0);

    if (addr != nullptr && state_->data_ != addr) {
        FATAL("%p vs %p", state_->data_, addr);
    }
    DEBUG("mmapped: %p (%zd)", state_->data_, state_->hostSize_);

    state_->data_   = reinterpret_cast<char *>(state_->data_) + offset;
    state_->offset_ = offset;
}


}
