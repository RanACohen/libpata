#include "pata_tensor.h"
#include "pata_errors.h"
#include "pata_operator.h"
#include "pata_stream.h"
#include "pata_backend.h"

using namespace libpata;
ScheduleTimeMeasurement libpata::schedule_time_map = {};

void libpata::schedule(const std::shared_ptr<ComputeCmd> &cmd)
{
    auto manager = BackendManager::Inst();
    auto stream = manager.backend()->createStream();
    for (auto in : cmd->inputs())
    {
        /* Since this command have input tensors that are not ready yet,
           we need to add dependecy "wait" for the current command. */ 
        CommandPtr wait = in.getWaitIfNotReady();
        if (wait)
        {                        
            stream->push(wait);
        }
    }

    for (auto out: cmd->outputs())
    {
        out.mark_not_ready();
    }

    stream->push(cmd);
    for (auto out: cmd->outputs())
    {
        stream->push(out.get_signal_cmd());
    }
    auto duration = chrono::high_resolution_clock::now().time_since_epoch();
    //schedule_time_map.push_back(std::chrono::duration_cast<std::chrono::microseconds>(duration));
}

KernelFunction::KernelFunction(const std::string &code)
{
}

KernelFunction::KernelFunction(const char *code)
{

}

void libpata::parallel_for(const Range &index, const KernelFunction &func)
{
}

Tensor libpata::reluN(const Tensor &in)
{
    Tensor out(in.shape(), in.dtype(), in.workspace());

    auto cmd = BackendManager::Inst().backend()->createComputeCmd("pata.reluN", {in}, {out}, {Attr(INT64, "max_int"), Attr(FLOAT, "max_fp")});
    schedule(cmd);

    return out;
}

Tensor libpata::abs(const Tensor &in)
{
    Tensor out(in.shape(), in.dtype(), in.workspace());
    
    auto cmd = BackendManager::Inst().backend()->createComputeCmd("pata.abs", {in}, {out}, {});
    schedule(cmd);
    return out;
}
