#include <thread>
#include <future>

#include "pata_debug.h"
#include "cpu_backend.h"
#include "cpu_commands.h"
//#include "pata_stream_pool.h"

#include "libxsmm.h"

using namespace libpata;
using namespace libpata::impl;

CPUBackend::CPUBackend()
{
    libxsmm_init();
}


void CPUBackend::wait_for_all() 
{
    return; // todo _pool->wait_for_all();
}

int CPUBackend::get_number_of_active_streams()
{
    return 0; // todo_pool->get_number_of_active_streams();
}

ComputeCmdPtr CPUBackend::createComputeCmd(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs)
{
    return std::make_shared<CPUComputeCmd>(op_name, inputs, outputs);
}

void CPUBackend::schedule(const std::shared_ptr<Wait>&wait_cmd, bool run_sync)
{
    if (wait_cmd->is_ready())
    {
        auto barrier = std::dynamic_pointer_cast<Barrier>(wait_cmd);
        if (barrier)
        {
            std::dynamic_pointer_cast<CPUBarrier>(barrier)->signal();
            return;
        }
        auto cmd = std::dynamic_pointer_cast<ComputeCmd>(wait_cmd->cmd_waiting());
        execute_cmd(cmd, run_sync);
    }

    // else we will check this again when the signal will be ready
}

void CPUBackend::execute_cmd(const ComputeCmdPtr &cmd, bool is_sync)
{    
    auto cpu_cmd = std::dynamic_pointer_cast<CPUCommand>(cmd);
    PATA_ASSERT(cpu_cmd && "only CPU commands can be executed...");
    if (is_sync)
    {
        //LOG() << "executing " << cmd->name() << ":" << cmd->id() << "\n";
        cpu_cmd->execute(this);        
        for (auto s: cmd->get_signals())
            std::dynamic_pointer_cast<CPUSignal>(s)->mark_ready(this);
        
        return;
    }
    //LOG() << "Scheduling  " << cmd->name() << "\n";
    std::thread{ [=]{
        static std::atomic<unsigned int> tid(1);
        set_local_thread_id(tid++);
        execute_cmd(cmd, true);
    }}.detach();    
}

std::shared_ptr<Barrier> CPUBackend::createBarrierCmd()
{
    return std::make_shared<CPUBarrier>();
}

std::shared_ptr<Signal> CPUBackend::createSignal()
{
    return std::make_shared<CPUSignal>();
}

std::shared_ptr<Wait> CPUBackend::createWait(const CommandPtr&cmd)
{
    return std::make_shared<Wait>(cmd);
}

CommandPtr CPUBackend::createTestCmd(int *variable, int test_val, int sleep_ms)
{
    return std::make_shared<TestCommand>(variable, test_val, sleep_ms);
}

ComputeCmdPtr CPUBackend::AddCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output)
{    
    return std::make_shared<CPUAddCmd>(lhs, rhs, output);
}

ComputeCmdPtr CPUBackend::MatMulCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output)
{
    return std::make_shared<CPUMatMulCmd>(lhs, rhs, output);
}
