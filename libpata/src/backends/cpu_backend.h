//
// Created by rcohen on 20/6/2021.
//
#pragma once
#ifndef LIBPATA_PATA_CPU_BACKEND_H
#define LIBPATA_PATA_CPU_BACKEND_H
#include <list>
#include <mutex>

#include "pata_backend.h"
#include "pata_threads.h"
#include "cpu/cpu_commands.h"

namespace libpata {
    namespace impl {
        class CPUBackend: public Backend {
            friend class libpata::BackendManager;            
            CPUBackend();                        
            void execute_cmd(const CommandPtr &cmd);
            std::list<CommandPtr> _pending_commands_lot;
            std::mutex            _command_ready_mutex;
            std::mutex            _log_mx;
            ObjectPool<CPUAddCmd> _add_cmd_pool;
        public:
            int get_number_of_active_streams(); // for later, counting active threads.
            virtual void wait_for_all();
            virtual void schedule(const CommandPtr &cmd);
            virtual std::shared_ptr<Signal> createSignal();            
            virtual std::shared_ptr<Barrier> createBarrierCmd();
            virtual ComputeCmdPtr createComputeCmd(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs);
            virtual CommandPtr createTestCmd(int *variable, int test_val, int sleep_ms);

            virtual ComputeCmdPtr AddCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output);
            virtual ComputeCmdPtr MatMulCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output);
        };
    }
}

#endif //LIBPATA_PATA_CPU_BACKEND_H
