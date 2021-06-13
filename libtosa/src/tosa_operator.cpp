#include "tosa_tensor.h"
#include "tosa_errors.h"
#include "tosa_operator.h"
#include "tosa_stream.h"
//#include "ThreadPool.h"

using namespace libtosa;


Operator::Operator(const std::shared_ptr<Operator> &op)
{
    _name = op->_name;
    _outputs = op->_outputs;
    _inputs = op->_inputs;
    _attributes = op->_attributes;
}
Operator::Operator(const Operator &base)
{
    _name = base._name;
    _outputs = base._outputs;
    _inputs = base._inputs;
    _attributes = base._attributes;
}

void libtosa::schedule(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs, const AttrList &attributes)
{
    auto manager = StreamManager::Inst();
    for (auto in : inputs)
    {
        /* Since this command have input tensors that are not ready yet,
           we need to add dependecy "wait" for the current command. */ 
        if (!in.is_ready())
        {
            OperatorPtr wait = std::make_shared<Operator>("wait");
            in.signal()->push_back(wait);
            manager.createStream()->add_single_op(wait);
        }
    }
    manager.createStream()->push(std::make_shared<Operator>(op_name, inputs, outputs, attributes));
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
