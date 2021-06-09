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

void libtosa::schedule(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs, const AttrList &attributes)
{
    /*static ThreadPool pool(POOL_SIZE);
    std::vector<std::future<int>> results;
    results.emplace_back(
        pool.enqueue([op_name]
                     {
                         std::cout << "running " << op_name << std::endl;
                         // todo: should be actually performing the operation, by taking the inputs + attributes,
                         //       compute and write output - and update it's "ready" bool member.
                         return 0;
                     }));
    */
   StreamManager manager = StreamManager::Inst();

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
