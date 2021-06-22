//
// Created by rcohen on 20/6/2021.
//
#pragma once
#ifndef LIBXLA_XLA_CPU_BACKEND_H
#define LIBXLA_XLA_CPU_BACKEND_H
#include <string>
#include <memory>
#include <condition_variable>

#include "xla_backend.h"
#include "xla_stream.h"
#include "xla_stream_pool.h"

namespace libxla {
    namespace impl {
        class CPUBackend: public Backend {
            friend class libxla::BackendManager;
            CPUBackend();
            std::mutex _pool_mutex;
            std::shared_ptr<libxla::StreamPool> _pool;

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

#endif //LIBXLA_XLA_CPU_BACKEND_H
