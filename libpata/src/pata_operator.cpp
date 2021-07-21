#include "libxsmm.h" // for test this is temporary

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
    auto wait = manager.backend()->createWait();
    for (auto in : cmd->inputs())
    {
        /* Since this command have input tensors that are not ready yet,
           we need to add dependecy "wait" for the current command. */ 
        
        in.getWaitList(wait);
    }
    if (!wait->is_empty())
    {
        stream->push(wait);
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
    auto out_rows = out.shape(0);
    auto out_cols = out.shape(1);
    auto common   = inA.shape(1);
    PATA_ASSERT(inA.rank()==2 && inB.rank()==2 && out.rank()==2);
    PATA_ASSERT(inA.shape(0) == out_rows && inB.shape(1) == out_cols);
    PATA_ASSERT(common == inB.shape()[0]);

    for (size_t row=0; row<out_rows; row += 256)
    {
        for (size_t col=0; col<out_cols; col += 256)
        {
            // auto tv = out[{Range(row, row+256), Range(col, col+256)}];
            outViews.push_back(out);
            // schedule(BackendManager::Inst().backend()->MatMulCmd(inA, inB, tv));
            schedule(BackendManager::Inst().backend()->MatMulCmd(inA, inB, out));
        }
    }
}

void libpata::Add2D(Tensor& inA, Tensor& inB, Tensor& out, TensorsList &outViews, int block_size)
{
    auto out_rows = out.shape(0);
    auto out_cols = out.shape(1);

    PATA_ASSERT(inA.rank()==2 && inB.rank()==2 && out.rank()==2);
    PATA_ASSERT(inA.shape(0) == out_rows && inA.shape(1) == out_cols);
    PATA_ASSERT(inB.shape(0) == out_rows && inB.shape(1) == out_cols);

    for (size_t row=0; row<out_rows; row += block_size)
    {
        auto inA_view = inA[{Range(row, row+block_size), Range(out_cols)}];
        auto inB_view = inB[{Range(row, row+block_size), Range(out_cols)}];
        auto tv = out[{Range(row, row+block_size), Range(out_cols)}];
        outViews.push_back(tv);
        schedule(BackendManager::Inst().backend()->AddCmd(inA_view, inB_view, tv));
    }
}

void libpata::Add(const Tensor& inA, const Tensor& inB, Tensor& out)
{
    schedule(BackendManager::Inst().backend()->AddCmd(inA, inB, out));    
}

bool libpata::test_Libxsmm(const Tensor& a, const Tensor& b, Tensor& out)
{
    float *pInA = a.at<float>(0,0);
    float *pInB = b.at<float>(0,0);
    float *pOutC = out.at<float>(0,0);

    libxsmm_blasint lda,ldb,ldc;
    lda = a.shape(0);
    ldb = b.shape(0);
    ldc = out.shape(0);
    libxsmm_blasint m,n,k;
    m = a.shape(0);
    n = b.shape(1);
    k = a.shape(1);
     
    libxsmm_mmfunction<float> xmm(LIBXSMM_MELTW_FLAG_FUSE_NONE, 
                                        m, n, k, // m,n,k
                                        lda, ldb, ldc, 1);
    std::cout << "Starting Matrix Multuiply\n";
    xmm(pInA, pInB, pOutC);
    std::cout << "Done\n";

    return true;
}