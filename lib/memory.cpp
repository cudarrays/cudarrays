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

#include <csignal>
#include <cstring>
#include <errno.h>
#include <stddef.h>
#include <sys/mman.h>

#include "cudarrays/common.hpp"
#include "cudarrays/memory.hpp"

#include "cudarrays/detail/utils/log.hpp"

namespace cudarrays {

using myptr = char *;

static handler_fn no_handler;

class handler_sigsegv {
    myptr begin_;
    size_t count_;
    handler_fn fn_;
    bool set_;

public:
    handler_sigsegv(myptr begin, size_t count) :
        begin_(begin),
        count_(count),
        fn_(nullptr),
        set_(false)
    {
    }

    bool operator()(bool b)
    {
        ASSERT(set_, "memory> Function handler not set");
        return fn_(b);
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

    void set_handler(const handler_fn &fn)
    {
        if (set_)
            DEBUG("memory> Overwriting handler");
        fn_ = fn;
        set_ = true;
    }

    void reset_handler()
    {
        if (fn_ != NULL) {
            fn_ = no_handler;
        }
        set_ = false;
    }
};

using map_handler = std::map<myptr, handler_sigsegv>;

map_handler handlers;

static struct sigaction defaultAction;

static const int Signum_{SIGSEGV};

void handler_sigsegv_main(int s, siginfo_t *info, void *ctx)
{
    mcontext_t *mCtx = &((ucontext_t *)ctx)->uc_mcontext;

    unsigned long isWrite = mCtx->gregs[REG_ERR] & 0x2;

    myptr addr = myptr(info->si_addr);

    if (!isWrite) DEBUG("memory> Read SIGSEGV for %p", addr);
    else          DEBUG("memory> Write SIGSEGV for %p", addr);

    bool resolved = false;

    map_handler::iterator it = handlers.upper_bound(addr);
    if (it != handlers.end() &&
        (addr >= it->second.start() &&
         addr <  it->second.end())) {
        resolved = it->second(isWrite);
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
protect_range(void *_addr, size_t count, const handler_fn &fn)
{
    uint64_t page = (uint64_t(_addr) >> 12);
    myptr align_addr = myptr(page << 12);
    myptr addr = myptr(_addr);

    map_handler::iterator it = handlers.upper_bound(addr);
    // Check for overlaps
    if (it != handlers.end() &&
        (addr >= it->second.start() &&
         addr <  it->second.end())) {

        it->second.set_handler(fn);

        DEBUG("memory> %p-%p -> NONE", addr, addr + count);
        int err = mprotect(align_addr, count, PROT_NONE);
        assert(err == 0);
    } else {
        FATAL("memory> Mapping %p NOT FOUND", addr);
    }
}

void
unprotect_range(void *_addr)
{
    uint64_t page = (uint64_t(_addr) >> 12);
    myptr align_addr = myptr(page << 12);
    myptr addr = myptr(_addr);

    map_handler::iterator it = handlers.upper_bound(addr);

    if (it != handlers.end() &&
        (addr >= it->second.start() &&
         addr <  it->second.end())) {
        int err = mprotect(align_addr, it->second.size(), PROT_READ | PROT_WRITE);
        DEBUG("memory> %p-%p -> RW", it->second.start(), it->second.end());
        assert(err == 0);
    } else {
        FATAL("memory> Mapping %p NOT FOUND", addr);
    }
}

void
register_range(void *_addr, size_t count)
{
    myptr addr = myptr(_addr);

    map_handler::iterator it = handlers.upper_bound(addr);

    if (it != handlers.end() &&
        (addr >= it->second.start() &&
         addr <  it->second.end())) {
        FATAL("memory> Mapping %p-%p overlaps with %p-%p", addr, addr + count, it->second.start(), it->second.end());
    } else {
        DEBUG("memory> REGISTERING mapping %p-%p", addr, addr + count);
        handlers.insert(map_handler::value_type(addr + count,
                                                handler_sigsegv(addr, count)));
    }
}

void
unregister_range(void *_addr)
{
    myptr addr = myptr(_addr);

    map_handler::iterator it = handlers.upper_bound(addr);

    if (it != handlers.end() &&
        (addr >= it->second.start() &&
         addr <  it->second.end())) {
        unprotect_range(addr);
        DEBUG("memory> Removing mapping %p-%p", it->second.start(), it->second.end());
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
    if (sigaction(Signum_, &defaultAction, NULL) < 0) {
        FATAL("memory> Error restoring SIGSEGV handler: %s", strerror(errno));
    } else {
        DEBUG("memory> Restore SIGSEGV handler");
    }
}

}

/* vim:set ft=cpp backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
