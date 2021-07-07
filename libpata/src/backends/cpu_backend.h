//
// Created by rcohen on 20/6/2021.
//
#pragma once
#ifndef LIBPATA_PATA_CPU_BACKEND_H
#define LIBPATA_PATA_CPU_BACKEND_H
#include <string>
#include <memory>
#include <condition_variable>

#include "pata_backend.h"
#include "pata_stream.h"
#include "pata_stream_pool.h"

namespace libpata {
    namespace impl {
        class CPUBackend: public Backend {
            friend class libpata::BackendManager;
            CPUBackend();
            std::mutex _pool_mutex;
            std::shared_ptr<libpata::StreamPool> _pool;

            public:
            virtual StreamPtr createStream();
            virtual int get_number_of_active_streams();
            virtual void wait_for_all();
            virtual std::shared_ptr<Signal> createSignal();
            virtual ComputeCmdPtr createComputeCmd(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs, const AttrList &attributes);
            virtual CommandPtr createTestCmd(int *variable, int test_val, int sleep_ms);

            virtual ComputeCmdPtr AddCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output);
            virtual ComputeCmdPtr MatMulCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output);
        };
    }
}

#endif //LIBPATA_PATA_CPU_BACKEND_H
