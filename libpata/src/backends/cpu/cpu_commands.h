//
// Created by rcohen on 20/6/2021.
//
#pragma once
#ifndef LIBPATA_PATA_CPU_COMMANDS_H
#define LIBPATA_PATA_CPU_COMMANDS_H
#include <condition_variable>

#include "pata_commands.h"
#include "pata_operator.h"
#include "pata_debug.h"

namespace libpata
{
    namespace impl
    {
        class CPUBackend;
        class CPUCommand
        {
             size_t _id;
        public:
            CPUCommand();
            inline size_t id() const { return _id; }
            virtual ~CPUCommand() = default;
            virtual void execute(CPUBackend *cpu_backend) = 0;

            volatile bool scheduled;
            std::atomic_flag executed = ATOMIC_FLAG_INIT;

        };
        typedef std::shared_ptr<CPUCommand> CPUCommandPtr;

        class CPUComputeCmd : public ComputeCmd, public CPUCommand
        {
        public:
            CPUComputeCmd(const std::string &name): ComputeCmd(name, {}, {})
            {}

            virtual void execute(CPUBackend *cpu_backend)
            {
                LOG() << " CPU Default (impl me): executing " << _cmd_name << std::endl;
            }

            void mark_complete(CPUBackend *backend); // protected under a mutex
        };

        class CPUBarrier: public Barrier, public CPUCommand
        {
            std::condition_variable _cv;
            bool _is_ready = false;
        public:
            CPUBarrier() = default;
            virtual void wait()
            {                
                std::unique_lock<std::mutex> lk(_wait_list_mx);
                scheduled = true; 
                if (_wait_on_signals.empty()) {
                    return;
                }
                _cv.wait(lk, [=] { return _is_ready;});
            }

            virtual void execute(CPUBackend *cpu_backend)
            {}
            
            void signal()
            {
                std::unique_lock<std::mutex> lk(_wait_list_mx);
                _is_ready = true;
                _cv.notify_all();
            }
        };

        class TestCommand : public CPUComputeCmd
        {
            int *_var;
            int _test_val;
            int _msec_sleep;

        public:
            TestCommand(int *variable, int test_val, int sleep_ms = 0):
                CPUComputeCmd("Test"),_var(variable), _test_val(test_val), _msec_sleep(sleep_ms) {}

            virtual void execute(CPUBackend *cpu_backend);
        };

        class CPUAddCmd: public CPUComputeCmd
        {
            public:
            CPUAddCmd():
                CPUComputeCmd("pata.add")
                {}

            void set_tensors(const Tensor &lhs, const Tensor &rhs, const Tensor &output)
            {
                _inputs.push_back(lhs);
                _inputs.push_back(rhs);
                _outputs.push_back(output);
            }
            void clear()
            {
                _inputs.clear();
                _outputs.clear();
                _out_signals.clear();
                scheduled = false;
                executed.clear();
                std::unique_lock<std::mutex> lk(_wait_list_mx);
                _wait_on_signals.clear();
                
            }
            virtual void execute(CPUBackend *cpu_backend);
        };

        class CPUMatMulCmd: public CPUComputeCmd
        {
            public:
            CPUMatMulCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output):
                CPUComputeCmd("pata.matmul")
                {
                    _inputs.push_back(lhs);
                    _inputs.push_back(rhs);
                    _outputs.push_back(output);
                }
            virtual void execute(CPUBackend *cpu_backend);
        };

    }
}

#endif // LIBPATA_PATA_CPU_COMMANDS_H