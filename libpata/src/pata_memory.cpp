//
// Created by rancohen on 24/5/2021.
//
#include "pata_memory.h"

using namespace libpata;

Workspace::Workspace(size_t size_in_bytes, MemoryBank bank) {
    _left_space = size_in_bytes;
    _bankType = bank;
}

void *Workspace::allocate(size_t size_in_bytes) {
    // todo: implement me using a normal heap
    std::lock_guard<std::mutex> guard(_heap_mutex);
    if (_left_space<size_in_bytes)
        return nullptr;
    _left_space -= size_in_bytes;
    return malloc(size_in_bytes);
}

void Workspace::free(void *ptr, size_t size) {
    std::lock_guard<std::mutex> guard(_heap_mutex);
    _left_space += size;
    ::free(ptr);
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

MemoryBlock::~MemoryBlock()
{
    _workspace->free(_ptr, _size);
}