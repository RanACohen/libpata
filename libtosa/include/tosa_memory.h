//
// Created by rancohen on 24/5/2021.
//
#pragma once
#ifndef LIBTOSA_TOSA_MEMORY_H
#define LIBTOSA_TOSA_MEMORY_H
#include <memory>

namespace libtosa {
    enum MemoryBank {
        CPU = 0, // CPU shared place
        DEVICE =1  // Device non-shared (copyable via DMA to CPU shared)
    };

    class MemoryBlock;
    typedef std::shared_ptr<MemoryBlock> MemoryBlockPtr;

    class Workspace {
        friend class MemoryBlock;
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
        static MemoryBlockPtr allocate(size_t size, const WorkspacePtr &workspace)
        {
            // todo: this is not neat, std::make_shared cannot be used due to private access.
            // make_shared is putting the ref count near the pointer, this method does not and can trash the cache
            return MemoryBlockPtr(new MemoryBlock(size, workspace));
        }
    };

}

#endif //LIBTOSA_TOSA_MEMORY_H
