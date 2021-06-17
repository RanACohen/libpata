#include "tosa_tensor.h"
#include "tosa_errors.h"
#include "tosa_operator.h"
#include "tosa_stream.h"
//#include "ThreadPool.h"

using namespace libtosa;


void libtosa::schedule(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs, const AttrList &attributes)
{
    auto manager = StreamManager::Inst();
    auto stream = manager.createStream();
    for (auto in : inputs)
    {
        /* Since this command have input tensors that are not ready yet,
           we need to add dependecy "wait" for the current command. */ 
        CommandPtr wait = in.getWaitIfNotReady();
        if (wait)
        {                        
            stream->push(wait);
        }
    }
    for (auto out: outputs)
    {
        out.mark_not_ready();
    }
    stream->push(std::make_shared<CPUComputeCmd>(op_name, inputs, outputs, attributes));
    for (auto out: outputs)
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

    schedule("tosa.reluN", {in}, {out}, {Attr(INT64, "max_int"), Attr(FLOAT, "max_fp")});
    return out;
}

Tensor libtosa::abs(const Tensor &in)
{
    Tensor out(in.shape(), in.dtype(), in.workspace());

    schedule("tosa.abs", {in}, {out}, {});
    return out;
}
