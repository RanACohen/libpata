//
// Created by rcohen on 20/6/2021.
//
#pragma once
#ifndef LIBPATA_PATA_CPU_COMMANDS_H
#define LIBPATA_PATA_CPU_COMMANDS_H
#include <condition_variable>

#include "pata_commands.h"
#include "pata_operator.h"

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
        };
        typedef std::shared_ptr<CPUCommand> CPUCommandPtr;

        class CPUComputeCmd : public ComputeCmd, public CPUCommand
        {
        public:
            CPUComputeCmd(const std::string &name,
                          const TensorsList &in,
                          const TensorsList &out): ComputeCmd(name, in, out)
            {}

            virtual void execute(CPUBackend *cpu_backend)
            {
                std::cout << " CPU Default (impl me): executing " << _cmd_name << std::endl;
            }
        };

        class CPUSignal : public Signal
        {           
        public:
            CPUSignal() {}
            void mark_ready(CPUBackend *cpu_backend);
        };

        class CPUBarrier: public Barrier
        {
            std::mutex _mx;
            std::condition_variable _cv;
            bool _is_ready = false;
        public:
            CPUBarrier() = default;
            virtual void wait()
            {
                std::unique_lock<std::mutex> lk(_mx);
                _cv.wait(lk, [=] { return _is_ready;});
            }
            
            void signal()
            {
                std::unique_lock<std::mutex> lk(_mx);
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
                CPUComputeCmd("Test", {}, {}),_var(variable), _test_val(test_val), _msec_sleep(sleep_ms) {}

            virtual void execute(CPUBackend *cpu_backend);
        };

        class CPUAddCmd: public CPUComputeCmd
        {
            public:
            CPUAddCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output):
                CPUComputeCmd("pata.add", TensorsList({lhs, rhs}), TensorsList({output}))
                {}
            virtual void execute(CPUBackend *cpu_backend);
        };

        class CPUMatMulCmd: public CPUComputeCmd
        {
            public:
            CPUMatMulCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output):
                CPUComputeCmd("pata.matmul", TensorsList({lhs, rhs}), TensorsList({output}))
                {}
            virtual void execute(CPUBackend *cpu_backend);
        };

    }
}

#endif // LIBPATA_PATA_CPU_COMMANDS_H