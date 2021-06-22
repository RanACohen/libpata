//
// Created by rancohen on 24/5/2021.
//
#pragma once
#ifndef LIBXLA_XLA_MEMORY_H
#define LIBXLA_XLA_MEMORY_H
#include <memory>
#include <mutex>

namespace libxla {
    enum MemoryBank {
        CPU = 0, // CPU shared place
        DEVICE =1  // Device non-shared (copyable via DMA to CPU shared)
    };

    class MemoryBlock;
    typedef std::shared_ptr<MemoryBlock> MemoryBlockPtr;

    class Workspace {
        friend class MemoryBlock;
        std::mutex _heap_mutex;
        MemoryBank _bankType;
        size_t _left_space; // todo: do a normal heap with page rounding to avoid fragmentation and force alignment
    public:
        explicit Workspace(size_t size_in_bytes, MemoryBank bank=CPU);


    protected:
        void *allocate(size_t size_in_bytes);
        void free(void *ptr, size_t size);
    };
    typedef std::shared_ptr<Workspace> WorkspacePtr;

    class MemoryBlock {
        void *_ptr;
        size_t _size;
        WorkspacePtr _workspace;
    private:
        MemoryBlock(size_t size, const WorkspacePtr &workspace);

    public:
        const WorkspacePtr &workspace() const { return _workspace; }
        void *ptr() const { return _ptr;}

        static MemoryBlockPtr allocate(size_t size, const WorkspacePtr &workspace)
        {
            // todo: this is not neat, std::make_shared cannot be used due to private access.
            // make_shared is putting the ref count near the pointer, this method does not and can trash the cache
            return MemoryBlockPtr(new MemoryBlock(size, workspace));
        }
    };

}

#endif //LIBXLA_XLA_MEMORY_H
