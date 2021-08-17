#include <thread>
#include <future>

#include "pata_debug.h"
#include "cpu_backend.h"
#include "cpu_commands.h"
#include "pata_threads.h"

//#include "pata_stream_pool.h"

#include "libxsmm.h"

using namespace libpata;
using namespace libpata::impl;

CPUBackend::CPUBackend():_add_cmd_pool(16384),_signal_cmd_pool(16384)
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
    return std::make_shared<CPUComputeCmd>(op_name);
}

void CPUBackend::schedule(const CommandPtr &cmd)
{
    //auto cpu_cmd = std::dynamic_pointer_cast<CPUCommand>(cmd);
    //PATA_ASSERT(cpu_cmd && "only CPU command scan be scuedhed with CPU backend");
    cmd->scheduled = true;
    if (cmd->is_ready())
    {            
        auto barrier = std::dynamic_pointer_cast<Barrier>(cmd);
        if (barrier)
        {
            std::dynamic_pointer_cast<CPUBarrier>(barrier)->signal();
            return;
        }
#ifdef DEADLOCK_DBG        
        {
            _log_mx.lock();
            LOG() << "scheduling cmd " << cmd->name() << ":" <<cmd->id() << "\n"<< FlushLog();
            _log_mx.unlock();
        }
#endif
        execute_cmd(cmd);
    } else {
        // nothing todo, cmd is saved in the dependent signals
#ifdef DEADLOCK_DBG
        {
            _log_mx.lock();
            LOG() << "cmd " << cmd->name() << ":" <<cmd->id() << " waiting...\n"<< FlushLog();
            _log_mx.unlock();
        }
#endif

    }

    // else we will check this again when the signal will be ready
}

void CPUBackend::execute_cmd(const CommandPtr &cmd)
{    
    auto cpu_cmd = std::dynamic_pointer_cast<CPUComputeCmd>(cmd);
    PATA_ASSERT(cpu_cmd && "only CPU commands can be executed...");
    //LOG() << "Scheduling  " << cmd->name() << "\n";
    global_thread_pool.ExecuteJob([=]{
        if (cpu_cmd->executed.test_and_set())
            {
                LOG() << "!!!!!!!!!! cmd double execution detected!!! " << cmd->name() << ":" <<cmd->id() << "\n" << FlushLog();
                return;
            }
#ifdef DEADLOCK_DBG
        {
            _log_mx.lock();
            LOG() << "executing cmd " << cmd->name() << ":" <<cmd->id() << "\n" << FlushLog();
            _log_mx.unlock();
        }
#endif
        cpu_cmd->execute(this);        
        _command_ready_mutex.lock();
        cpu_cmd->mark_complete(this);
        _command_ready_mutex.unlock();                
    });
}

std::shared_ptr<Barrier> CPUBackend::createBarrierCmd()
{
    return std::make_shared<CPUBarrier>();
}

std::shared_ptr<Signal> CPUBackend::createSignal()
{
    return _signal_cmd_pool.get();
    //return std::make_shared<Signal>();
}


CommandPtr CPUBackend::createTestCmd(int *variable, int test_val, int sleep_ms)
{
    return std::make_shared<TestCommand>(variable, test_val, sleep_ms);
}

ComputeCmdPtr CPUBackend::AddCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output)
{    
    auto obj = _add_cmd_pool.get();
    obj->set_tensors(lhs, rhs, output);
    return obj;
}

ComputeCmdPtr CPUBackend::MatMulCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output)
{
    return std::make_shared<CPUMatMulCmd>(lhs, rhs, output);
}
