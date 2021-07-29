//
// Created by rcohen on 20/6/2021.
//
#pragma once
#ifndef LIBPATA_PATA_CPU_BACKEND_H
#define LIBPATA_PATA_CPU_BACKEND_H
#include <string>
#include <memory>
#include <thread>
#include <condition_variable>

#include "pata_backend.h"
#include "cpu/cpu_commands.h"

namespace libpata {
    namespace impl {
        class CPUBackend: public Backend {
            friend class libpata::BackendManager;
            friend class libpata::impl::CPUSignal;
            CPUBackend();                        
            void execute_cmd(const ComputeCmdPtr &cmd, bool is_sync);
        
        public:
            int get_number_of_active_streams(); // for later, counting active threads.
            virtual void wait_for_all();
            virtual void schedule(const std::shared_ptr<Wait>&wait_cmd, bool run_sync=false);
            virtual std::shared_ptr<Signal> createSignal();
            virtual std::shared_ptr<Wait> createWait(const CommandPtr&cmd);
            virtual std::shared_ptr<Barrier> createBarrierCmd();
            virtual ComputeCmdPtr createComputeCmd(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs);
            virtual CommandPtr createTestCmd(int *variable, int test_val, int sleep_ms);

            virtual ComputeCmdPtr AddCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output);
            virtual ComputeCmdPtr MatMulCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output);
        };
    }
}

#endif //LIBPATA_PATA_CPU_BACKEND_H
