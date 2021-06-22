#include <unistd.h>
#include "xla_types.h"
#include "cpu_commands.h"
#include "libxsmm.h"

using namespace libxla;
using namespace libxla::impl;

void CPUWait::execute()
{
    Signal *ps = _wait_on.get();
    auto sig = dynamic_cast<CPUSignal *>(ps);
    sig->wait();
}

void CPUSignal::wait()
{
    std::unique_lock<std::mutex> lk(_mutex);
    _cv.wait(lk, [=]
             { return is_ready(); });
}
void CPUSignal::execute()
{
    signal();
    _cv.notify_all();
}

std::shared_ptr<Wait> CPUSignal::getWaitCmd()
{
    return std::make_shared<CPUWait>(std::dynamic_pointer_cast<Signal>(shared_from_this()));
}

void TestCommand::execute()
{
    usleep(_msec_sleep * 1000);
    *_var = _test_val;
}

libxsmm_datatype xla_to_xsmm_dtype(DType dtype)
{
     return dtype == libxla::FLOAT ? LIBXSMM_DATATYPE_F32 :
                              dtype == libxla::BF16 ? LIBXSMM_DATATYPE_BF16 :
                              dtype == libxla::FP16 ? LIBXSMM_DATATYPE_F16 :
                              dtype == libxla::INT32 ? LIBXSMM_DATATYPE_I32 :
                              LIBXSMM_DATATYPE_UNSUPPORTED; // todo: throw here
}

void CPUAddCmd::execute()
{    
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
        libxsmm_datatype dt = dtype == libxla::FLOAT ? LIBXSMM_DATATYPE_F32 :
                              dtype == libxla::BF16 ? LIBXSMM_DATATYPE_BF16 :
                              dtype == libxla::FP16 ? LIBXSMM_DATATYPE_F16 :
                              dtype == libxla::INT32 ? LIBXSMM_DATATYPE_I32 :
                              LIBXSMM_DATATYPE_UNSUPPORTED; // todo: throw here

        XLA_ASSERT(dt != LIBXSMM_DATATYPE_UNSUPPORTED);

        libxsmm_meltwfunction_binary binary_kernel = libxsmm_dispatch_meltw_binary(shape[1], shape[0], 
            &ldi0, &ldi1, &ldo, 
            dt, dt, dt, binary_flags, binary_type);
        XLA_ASSERT((binary_kernel != NULL) && "JIT for BINARY TPP. Bailing...!");
        
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
        libxsmm_datatype dt = dtype == libxla::FLOAT ? LIBXSMM_DATATYPE_F32 :
                              dtype == libxla::BF16 ? LIBXSMM_DATATYPE_BF16 :
                              dtype == libxla::FP16 ? LIBXSMM_DATATYPE_F16 :
                              dtype == libxla::INT32 ? LIBXSMM_DATATYPE_I32 :
                              LIBXSMM_DATATYPE_UNSUPPORTED; // todo: throw here

        XLA_ASSERT(dt != LIBXSMM_DATATYPE_UNSUPPORTED);

        libxsmm_meltwfunction_binary binary_kernel = libxsmm_dispatch_meltw_binary(1, _inputs[0].volume(), &ldi, &ldi, &ldi, 
            dt, dt, dt, binary_flags, binary_type);
        XLA_ASSERT((binary_kernel != NULL) && "JIT for BINARY TPP. Bailing...!");
        
        binary_kernel(&binary_param);
        return;
    }
}