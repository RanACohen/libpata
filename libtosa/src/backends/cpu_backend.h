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
            virtual std::shared_ptr<Signal> createSignal();
            virtual CommandPtr createComputeCmd(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs, const AttrList &attributes);
            virtual CommandPtr createTestCmd(int *variable, int test_val, int sleep_ms);
        };
    }
}

#endif //LIBTOSA_TOSA_CPU_BACKEND_H
