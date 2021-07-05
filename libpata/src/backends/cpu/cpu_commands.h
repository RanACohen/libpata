//
// Created by rcohen on 20/6/2021.
//
#pragma once
#ifndef LIBPATA_PATA_CPU_COMMANDS_H
#define LIBPATA_PATA_CPU_COMMANDS_H

#include "pata_commands.h"
#include "pata_operator.h"

namespace libpata
{
    namespace impl
    {

        class CPUCommand : virtual public Command
        {
             size_t _id;
        public:
            CPUCommand();
            inline size_t id() const { return _id; }
            virtual ~CPUCommand() = default;
            virtual void execute(Stream *in_stream) = 0;
        };

        class CPUComputeCmd : virtual public ComputeCmd, CPUCommand
        {
        public:
            CPUComputeCmd(const std::string &name,
                          const TensorsList &in,
                          const TensorsList &out,
                          const AttrList &attr): ComputeCmd(name, in, out, attr) 
            {}

            virtual void execute(Stream *in_stream)
            {
                std::cout << " excuting " << _name << " in stream id " << in_stream->id() << std::endl;
            }
        };

        class CPUSignal : virtual public Signal, CPUCommand
        {           
            std::condition_variable _cv;
            std::mutex _mutex;      
            std::atomic<int> _ready;
            std::shared_ptr<TensorImpl> _orig_tensor;

        public:
            CPUSignal(std::shared_ptr<TensorImpl> t): _ready (0), _orig_tensor(t) {}
            void wait(Stream *wait_in_stream);
            virtual void execute(Stream *in_stream);
            virtual std::shared_ptr<Wait> getWaitCmd();
            virtual void mark_not_ready() { _ready++;}
            inline virtual void mark_ready()
            {
                if (!is_ready()) _ready--;
                _cv.notify_all();
            }
            inline bool is_ready() { return _ready == 0; }
        };

        class CPUWait : virtual public Wait, CPUCommand
        {
        public:
            CPUWait(const std::shared_ptr<Signal> &wait_on) : Wait(wait_on) {}
            virtual void execute(Stream *in_stream);
        };

        class TestCommand : virtual public Command, CPUCommand
        {
            int *_var;
            int _test_val;
            int _msec_sleep;

        public:
            TestCommand(int *variable, int test_val, int sleep_ms = 0) : _var(variable), _test_val(test_val), _msec_sleep(sleep_ms) {}

            virtual void execute(Stream *in_stream);
        };

        class CPUAddCmd: virtual public ComputeCmd, CPUCommand
        {
            public:
            CPUAddCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output):
                ComputeCmd("pata.add", TensorsList({lhs, rhs}), TensorsList({output}), AttrList({}))
                {}
            virtual void execute(Stream *in_stream);
        };

        class CPUMatMulCmd: virtual public ComputeCmd, CPUCommand
        {
            public:
            CPUMatMulCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output):
                ComputeCmd("pata.matmul", TensorsList({lhs, rhs}), TensorsList({output}), AttrList({}))
                {}
            virtual void execute(Stream *in_stream);
        };

    }
}

#endif // LIBPATA_PATA_CPU_COMMANDS_H