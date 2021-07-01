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

/**
     * MatMul: does a Matrix multiplication of out=A*B, 
     * inA - Input A Matrix, must be a 2D Tensor (todo: Batched MatMul for 3D+)
     * inB - Input B Matrix, must be a 2D Tensor
     * out - output Matrix (already allocated) of size (A-rows,B-cols)
     * outViews - an empty Tensor list of views to be placed after the split according to a h/w friendly split
     * */
void libpata::MatMul(const Tensor& inA, const Tensor& inB, Tensor& out, TensorsList &outViews)
{
    auto out_rows = out.shape()[0];
    auto out_cols = out.shape()[1];
    auto common   = inA.shape()[1];
    PATA_ASSERT(inA.rank()==2 && inB.rank()==2 && out.rank()==2);
    PATA_ASSERT(inA.shape()[0] == out_rows && inB.shape()[1]==out_cols);
    PATA_ASSERT(common == inB.shape()[0]);

    if (out_rows>256 || out_cols>256)
    {
        for (size_t row=0; row<out_rows; row += 256)
        {
            for (size_t col=0; col<out_cols; col += 256)
            {
                auto tv = out[{Range(row, row+256), Range(col, col+256)}];
                outViews.push_back(tv);
                schedule(BackendManager::Inst().backend()->MatMulCmd(inA, inB, tv));
            }
        }
    }
}