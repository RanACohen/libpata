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
#include "tosa_stream.h"
#include "tosa_stream_pool.h"

namespace libtosa {
    namespace impl {
        class CPUBackend: public Backend {
            friend class libtosa::BackendManager;
            CPUBackend();
            std::mutex _pool_mutex;
            std::shared_ptr<libtosa::StreamPool> _pool;

            public:
            virtual StreamPtr createStream();
            virtual void wait_for_all();
            virtual std::shared_ptr<Signal> createSignal();
            virtual ComputeCmdPtr createComputeCmd(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs, const AttrList &attributes);
            virtual CommandPtr createTestCmd(int *variable, int test_val, int sleep_ms);

            virtual ComputeCmdPtr AddCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output);
        };
    }
}

#endif //LIBTOSA_TOSA_CPU_BACKEND_H
