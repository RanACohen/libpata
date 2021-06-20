#include "tosa_tensor.h"
#include "tosa_errors.h"
#include "tosa_operator.h"
#include "tosa_stream.h"
#include "tosa_backend.h"


using namespace libtosa;


void libtosa::schedule(const std::shared_ptr<ComputeCmd> &cmd)
{
    auto manager = StreamManager::Inst();
    auto stream = manager.createStream();
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
}

KernelFunction::KernelFunction(const std::string &code)
{
}

KernelFunction::KernelFunction(const char *code)
{

}

void libtosa::parallel_for(const Range &index, const KernelFunction &func)
{

}


Tensor libtosa::reluN(const Tensor &in)
{
    Tensor out(in.shape(), in.dtype(), in.workspace());

    auto cmd = BackendManager::Inst().backend()->createComputeCmd("tosa.reluN", {in}, {out}, {Attr(INT64, "max_int"), Attr(FLOAT, "max_fp")});
    schedule(cmd);

    return out;
}

Tensor libtosa::abs(const Tensor &in)
{
    Tensor out(in.shape(), in.dtype(), in.workspace());
    
    auto cmd = BackendManager::Inst().backend()->createComputeCmd("tosa.abs", {in}, {out}, {});
    schedule(cmd);
    return out;
}
