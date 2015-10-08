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

#include <map>
#include <memory>

#include <csignal>
#include <cstring>
#include <errno.h>
#include <stddef.h>
#include <sys/mman.h>

#include "cudarrays/common.hpp"
#include "cudarrays/memory.hpp"

#include "cudarrays/detail/utils/option.hpp"
#include "cudarrays/detail/utils/log.hpp"

namespace cudarrays {

static inline int
mem_access_to_prot(mem_access_type access)
{
    switch (access) {
    case MEM_NONE:       return PROT_NONE;
    case MEM_READ:       return PROT_READ;
    case MEM_WRITE:      return PROT_WRITE;
    case MEM_READ_WRITE: return PROT_READ | PROT_WRITE;
    };
    FATAL("Invalid access type value");
    return 0;
}

std::string
to_string(mem_access_type access_type)
{
    switch (access_type) {
    case MEM_NONE:       return "NONE";
    case MEM_READ:       return "READ";
    case MEM_WRITE:      return "WRITE";
    case MEM_READ_WRITE: return "READ_WRITE";
    default:             abort();
    };
}

using myptr = char *;

class handler_sigsegv {
    myptr begin_;
    size_t count_;
    mem_access_type prot_;
    std::shared_ptr<handler_fn> fn_;
    bool set_;

public:
    handler_sigsegv(myptr begin, size_t count, mem_access_type prot) :
        begin_(begin),
        count_(count),
        prot_(prot),
        fn_(nullptr),
        set_(false)
    {
    }

    bool operator()(bool b)
    {
        ASSERT(set_, "memory> Function handler not set");
        std::shared_ptr<handler_fn> fn = fn_;
        return (*fn)(b);
    }

    myptr start()
    {
        return begin_;
    }

    myptr end()
    {
        return begin_ + count_;
    }

    size_t size() const
    {
        return count_;
    }

    mem_access_type protection() const
    {
        return prot_;
    }

    void set_protection(mem_access_type prot)
    {
        prot_ = prot;
    }

    void set_handler(const handler_fn &fn)
    {
        if (set_)
            DEBUG("memory> Overwriting handler");
        auto ptr = std::make_shared<handler_fn>(fn);
        fn_.swap(ptr);
        set_ = true;
    }
};

using map_handler = std::map<myptr, handler_sigsegv>;

map_handler handlers;

static struct sigaction defaultAction;

static const int Signum_{SIGSEGV};

void handler_sigsegv_main(int s, siginfo_t *info, void *ctx)
{
    TRACE_FUNCTION();

    mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;

    unsigned long isWrite = mCtx->gregs[REG_ERR] & 0x2;

    myptr addr = myptr(info->si_addr);

    if (!isWrite) DEBUG("memory> Read SIGSEGV for %p", addr);
    else          DEBUG("memory> Write SIGSEGV for %p", addr);

    bool resolved = false;

    map_handler::iterator it = handlers.upper_bound(addr);
    if (it == handlers.end())
        FATAL("memory> Mapping %p NOT FOUND", addr);

    handler_sigsegv &handler = it->second;

    if (addr >= handler.start() &&
        addr <  handler.end()) {
        resolved = handler(isWrite);
    }

    if (resolved == false) {
        DEBUG("memory> Uoops! I could not find a mapping for %p. Forwarding it to the OS.", addr);

        // TODO: set the signal mask and other stuff
        if (defaultAction.sa_flags & SA_SIGINFO)
            return defaultAction.sa_sigaction(s, info, ctx);
        return defaultAction.sa_handler(s);
    }
}

void
protect_range(void *_addr, size_t count, mem_access_type access_type, handler_fn fn)
{
    TRACE_FUNCTION();

    uint64_t page = (uint64_t(_addr) >> 12);
    myptr align_addr = myptr(page << 12);
    myptr addr = myptr(_addr);

    map_handler::iterator it = handlers.upper_bound(addr);

    if (it == handlers.end())
        FATAL("memory> Mapping %p NOT FOUND", addr);

    handler_sigsegv &handler = it->second;

    // Check for overlaps
    ASSERT(addr >= handler.start() && addr < handler.end(),
           "memory> Invalid mapping for %p", addr);

    handler.set_handler(fn);

    if (access_type == handler.protection())
        return;

    handler.set_protection(access_type);

    int err = mprotect(align_addr, count, mem_access_to_prot(access_type));
    DEBUG("memory> %p-%p -> %s", addr, addr + count,
          to_string(access_type));
    assert(err == 0);
}

void
register_range(void *_addr, size_t count)
{
    TRACE_FUNCTION();

    myptr addr = myptr(_addr);

    map_handler::iterator it = handlers.upper_bound(addr);
    if (it != handlers.end() &&
        (addr >= it->second.start() &&
         addr <  it->second.end())) {
        FATAL("memory> Mapping %p-%p overlaps with %p-%p", addr, addr + count, it->second.start(), it->second.end());
    } else {
        DEBUG("memory> REGISTERING mapping %p-%p", addr, addr + count);
        handlers.insert(map_handler::value_type(addr + count,
                                                handler_sigsegv{addr, count, MEM_READ_WRITE}));
    }
}

void
unregister_range(void *_addr)
{
    TRACE_FUNCTION();

    myptr addr = myptr(_addr);

    map_handler::iterator it = handlers.upper_bound(addr);
    if (it == handlers.end())
        FATAL("memory> Mapping %p NOT FOUND", addr);

    handler_sigsegv &handler = it->second;
    if (addr >= handler.start() && addr < handler.end()) {
        protect_range(handler.start(), handler.size(), mem_access_type::MEM_READ_WRITE);
        DEBUG("memory> Removing mapping %p-%p", handler.start(), handler.end());

        handlers.erase(it);
    } else {
        // Not found!
        // FATAL("Mapping %pp", it->second.start(), it->second.end());
        //abort();
    }
}

void
handler_sigsegv_overload()
{
    TRACE_FUNCTION();

    struct sigaction segvAction;
    ::memset(&segvAction, 0, sizeof(segvAction));
    segvAction.sa_sigaction = handler_sigsegv_main;
    segvAction.sa_flags = SA_SIGINFO | SA_RESTART;
    sigemptyset(&segvAction.sa_mask);

    if (sigaction(Signum_, &segvAction, &defaultAction) < 0) {
        FATAL("memory> Error installing SIGSEGV handler: %s", strerror(errno));
    } else {
        DEBUG("memory> Install SIGSEGV handler");
    }
}

void
handler_sigsegv_restore()
{
    TRACE_FUNCTION();

    if (sigaction(Signum_, &defaultAction, NULL) < 0) {
        FATAL("memory> Error restoring SIGSEGV handler: %s", strerror(errno));
    } else {
        DEBUG("memory> Restore SIGSEGV handler");
    }
}

}

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
