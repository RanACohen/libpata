#include "xla_tensor.h"
#include "xla_errors.h"
#include "xla_operator.h"
#include "xla_stream.h"
#include "xla_backend.h"

using namespace libxla;
ScheduleTimeMeasurement libxla::schedule_time_map = {};

void libxla::schedule(const std::shared_ptr<ComputeCmd> &cmd)
{
    auto start_time = chrono::high_resolution_clock::now();
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
    auto end_time = chrono::high_resolution_clock::now();
    //schedule_time_map[cmd->name()] = chrono::duration_cast<chrono::microseconds>(end_time - start_time);
}

KernelFunction::KernelFunction(const std::string &code)
{
}

KernelFunction::KernelFunction(const char *code)
{

}

void libxla::parallel_for(const Range &index, const KernelFunction &func)
{
}

Tensor libxla::reluN(const Tensor &in)
{
    Tensor out(in.shape(), in.dtype(), in.workspace());

    auto cmd = BackendManager::Inst().backend()->createComputeCmd("xla.reluN", {in}, {out}, {Attr(INT64, "max_int"), Attr(FLOAT, "max_fp")});
    schedule(cmd);

    return out;
}

Tensor libxla::abs(const Tensor &in)
{
    Tensor out(in.shape(), in.dtype(), in.workspace());
    
    auto cmd = BackendManager::Inst().backend()->createComputeCmd("xla.abs", {in}, {out}, {});
    schedule(cmd);
    return out;
}
