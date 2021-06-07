#include "tosa_tensor.h"
#include "tosa_errors.h"
#include "tosa_operator.h"

using namespace libtosa;

void libtosa::schedule(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs, const AttrList &attributes)
{
    static ThreadPool pool(POOL_SIZE);
    std::vector<std::future<int>> results;
    results.emplace_back(
        pool.enqueue([op_name]
                     {
                         std::cout << "running " << op_name << std::endl;
                         // todo: should be actually performing the operation, by taking the inputs + attributes,
                         //       compute and write output - and update it's "ready" bool member.
                         return 0;
                     }));
}

KernelFunction::KernelFunction(const std::string &code)
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
