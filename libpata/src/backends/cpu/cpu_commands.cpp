#include <atomic>

#include <unistd.h>
#include "pata_types.h"
#include "cpu_commands.h"
#include "libxsmm.h"
#include "pata_debug.h"

using namespace libpata;
using namespace libpata::impl;


void CPUWait::execute(Stream *in_stream)
{
    Signal *ps = _wait_on.get();
    auto sig = dynamic_cast<CPUSignal *>(ps);
    sig->wait(in_stream);
}

CPUCommand::CPUCommand()
{
    static std::atomic<size_t> gid(0);
    _id = gid++;
}

void CPUSignal::wait(Stream *wait_in_stream)
{
    std::unique_lock<std::mutex> lk(_mutex);
        
    if (is_ready())
    {
        log_dead_lock(wait_in_stream->id(), id(), -1, EventType::STREAM_WAIT);
    } else {
        log_dead_lock(wait_in_stream->id(), id(), sched_in_stream->id(), EventType::STREAM_WAIT);
    }    
    _cv.wait(lk, [=]{ return is_ready(); });
    log_dead_lock(wait_in_stream->id(), id(), -1, EventType::WAIT_WOKE_UP);
}

void CPUSignal::execute(Stream *in_stream)
{
    std::unique_lock<std::mutex> lk(_mutex);
    log_dead_lock(in_stream->id(), id(), -1, EventType::SIGNAL_ON);
    signal();
    _cv.notify_all();
}

std::shared_ptr<Wait> CPUSignal::getWaitCmd()
{
    return std::make_shared<CPUWait>(std::dynamic_pointer_cast<Signal>(shared_from_this()));
}

void TestCommand::execute(Stream *in_stream)
{
    usleep(_msec_sleep * 1000);
    *_var = _test_val;
}

libxsmm_datatype pata_to_xsmm_dtype(DType dtype)
{
     return dtype == libpata::FLOAT ? LIBXSMM_DATATYPE_F32 :
                              dtype == libpata::BF16 ? LIBXSMM_DATATYPE_BF16 :
                              dtype == libpata::FP16 ? LIBXSMM_DATATYPE_F16 :
                              dtype == libpata::INT32 ? LIBXSMM_DATATYPE_I32 :
                              LIBXSMM_DATATYPE_UNSUPPORTED; // todo: throw here
}

void CPUAddCmd::execute(Stream *in_stream)
{    
    return;
    if (_inputs[0].rank() == 2)
    {
        libxsmm_meltw_binary_param binary_param;
        libxsmm_meltw_binary_flags binary_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
        libxsmm_meltw_binary_type binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;

        auto shape = _inputs[0].shape();

        binary_param.in0.primary = _inputs[0].base_addr();
        binary_param.in1.primary = _inputs[1].base_addr();
        binary_param.out.primary = _outputs[0].base_addr();
        libxsmm_blasint ldi0 = _inputs[0].stride()[0];
        libxsmm_blasint ldi1 = _inputs[1].stride()[0];
        libxsmm_blasint ldo = _outputs[0].stride()[0];
        auto dtype = _inputs[0].dtype();
        libxsmm_datatype dt = dtype == libpata::FLOAT ? LIBXSMM_DATATYPE_F32 :
                              dtype == libpata::BF16 ? LIBXSMM_DATATYPE_BF16 :
                              dtype == libpata::FP16 ? LIBXSMM_DATATYPE_F16 :
                              dtype == libpata::INT32 ? LIBXSMM_DATATYPE_I32 :
                              LIBXSMM_DATATYPE_UNSUPPORTED; // todo: throw here

        PATA_ASSERT(dt != LIBXSMM_DATATYPE_UNSUPPORTED);

        libxsmm_meltwfunction_binary binary_kernel = libxsmm_dispatch_meltw_binary(shape[1], shape[0], 
            &ldi0, &ldi1, &ldo, 
            dt, dt, dt, binary_flags, binary_type);
        PATA_ASSERT((binary_kernel != NULL) && "JIT for BINARY TPP. Bailing...!");
        
        binary_kernel(&binary_param);
        return;
    }

    if (_inputs[0].is_contiguous() && _inputs[1].is_contiguous() && _outputs[0].is_contiguous())
    {
        libxsmm_meltw_binary_param binary_param;
        libxsmm_meltw_binary_flags binary_flags = LIBXSMM_MELTW_FLAG_BINARY_NONE;
        libxsmm_meltw_binary_type binary_type = LIBXSMM_MELTW_TYPE_BINARY_ADD;

        binary_param.in0.primary = _inputs[0].base_addr();
        binary_param.in1.primary = _inputs[1].base_addr();
        binary_param.out.primary = _outputs[0].base_addr();
        libxsmm_blasint ldi = _inputs[0].volume();
        auto dtype = _inputs[0].dtype();
        libxsmm_datatype dt = dtype == libpata::FLOAT ? LIBXSMM_DATATYPE_F32 :
                              dtype == libpata::BF16 ? LIBXSMM_DATATYPE_BF16 :
                              dtype == libpata::FP16 ? LIBXSMM_DATATYPE_F16 :
                              dtype == libpata::INT32 ? LIBXSMM_DATATYPE_I32 :
                              LIBXSMM_DATATYPE_UNSUPPORTED; // todo: throw here

        PATA_ASSERT(dt != LIBXSMM_DATATYPE_UNSUPPORTED);

        libxsmm_meltwfunction_binary binary_kernel = libxsmm_dispatch_meltw_binary(_inputs[0].volume(), 1, &ldi, &ldi, &ldi, 
            dt, dt, dt, binary_flags, binary_type);
        PATA_ASSERT((binary_kernel != NULL) && "JIT for BINARY TPP. Bailing...!");
        
        binary_kernel(&binary_param);
        return;
    }
}