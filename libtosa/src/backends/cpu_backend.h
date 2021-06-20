//
// Created by rcohen on 20/6/2021.
//
#pragma once
#ifndef LIBTOSA_TOSA_CPU_BACKEND_H
#define LIBTOSA_TOSA_CPU_BACKEND_H
#include <string>
#include <memory>
#include <condition_variable>

#include "tosa_backend.h"

namespace libtosa {
    namespace impl {
        class CPUBackend: public Backend {
            friend class libtosa::BackendManager;
            CPUBackend() = default;
            public:
            virtual Stream *createStream(int id);
        };
    }
}

#endif //LIBTOSA_TOSA_CPU_BACKEND_H
