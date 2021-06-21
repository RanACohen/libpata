#include "tosa_backend.h"
#include "backends/cpu_backend.h"

using namespace libtosa;

BackendManager &BackendManager::Inst()
{
    static BackendManager inst;

    return inst;
}


BackendManager::BackendManager()
{
    _backends[CPU] = new impl::CPUBackend();
    _backends[GAUDI] = nullptr; // lazy creation in set
    _active_backend = _backends[CPU];
}


void BackendManager::set_backend(BACKEND_TYPE type)
{
    // todo: makre reentrent
    _active_backend = _backends[type];
    if (_backends[type]!=nullptr) return;

    switch (type)    
    {
    case CPU:
        throw std::runtime_error("should not happen, CPU is created my default.");
        break;
    case GAUDI:
        throw std::runtime_error("Not implemented yet");

    default:    
        throw std::runtime_error("unrecognized backend type");
        break;
    }
}

