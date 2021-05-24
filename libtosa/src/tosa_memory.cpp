//
// Created by rancohen on 24/5/2021.
//
#include "tosa_memory.h"

using namespace libtosa;

Workspace::Workspace(size_t size_in_bytes) {

}

void *Workspace::allocate(size_t size_in_bytes) {
    // todo: implement me
    return nullptr;
}

void Workspace::free(void *ptr, size_t size) {
    // todo: implement me
}

/**
 * MemoryBlock
 * @param ptr
 * @param size
 * @param workspace
 */
MemoryBlock::MemoryBlock(size_t size, const WorkspacePtr &workspace) {
    _workspace = workspace;
    _size = size;
    _ptr = workspace->allocate(size);
}
