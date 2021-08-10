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

void CPUBackend::schedule(const CommandPtr &cmd)
{
    if (cmd->is_ready())
    {            
        auto barrier = std::dynamic_pointer_cast<Barrier>(cmd);
        if (barrier)
        {
            std::dynamic_pointer_cast<CPUBarrier>(barrier)->signal();
            return;
        }
        execute_cmd(cmd);
    } else {
        // todo: protect with mutext? 
        cmd->_pending_location = _pending_commands_lot.insert(_pending_commands_lot.begin(), cmd);
    }

    // else we will check this again when the signal will be ready
}

void CPUBackend::execute_cmd(const CommandPtr &cmd)
{    
    auto cpu_cmd = std::dynamic_pointer_cast<CPUComputeCmd>(cmd);
    PATA_ASSERT(cpu_cmd && "only CPU commands can be executed...");
    //LOG() << "Scheduling  " << cmd->name() << "\n";
    std::thread{ [=]{
        static std::atomic<unsigned int> tid(1);
        set_local_thread_id(tid++);
        cpu_cmd->execute(this);        
        _command_ready_mutex.lock();
        cpu_cmd->mark_complete(this);
        _command_ready_mutex.unlock();
                
    }}.detach();    
}

std::shared_ptr<Barrier> CPUBackend::createBarrierCmd()
{
    return std::make_shared<CPUBarrier>();
}

std::shared_ptr<Signal> CPUBackend::createSignal()
{
    return std::make_shared<Signal>();
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
