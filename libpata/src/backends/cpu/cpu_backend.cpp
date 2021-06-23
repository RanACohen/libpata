#include "cpu_backend.h"
#include "cpu_commands.h"
#include "cpu_stream.h"
#include "pata_stream_pool.h"

#include "libxsmm.h"

using namespace libpata;
using namespace libpata::impl;

CPUBackend::CPUBackend()
{
    libxsmm_init();
}

StreamPtr CPUBackend::createStream() 
{
    if (!_pool)
    {
        std::lock_guard<std::mutex> guard(_pool_mutex);
        if (!_pool)
            _pool = std::make_shared<libpata::StreamPool>(5, [](int i)-> Stream* {return new CPUStream(i); });
    }
            
    return _pool->createStream();
}

void CPUBackend::wait_for_all() 
{
    return _pool->wait_for_all();
}

ComputeCmdPtr CPUBackend::createComputeCmd(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs, const AttrList &attributes)
{
    return std::make_shared<CPUComputeCmd>(op_name, inputs, outputs, attributes);
}

std::shared_ptr<Signal> CPUBackend::createSignal()
{
    return std::make_shared<CPUSignal>();
}

CommandPtr CPUBackend::createTestCmd(int *variable, int test_val, int sleep_ms)
{
    return std::make_shared<TestCommand>(variable, test_val, sleep_ms);
}

ComputeCmdPtr CPUBackend::AddCmd(const Tensor &lhs, const Tensor &rhs, const Tensor &output)
{
    // todo: put special AddCmd
    auto c = new CPUAddCmd(lhs, rhs, output);
    return std::make_shared<CPUAddCmd>(lhs, rhs, output);
}


