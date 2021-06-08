#include <iostream>
#include <gtest/gtest.h>

#include "tosa_tensor.h"
#include "tosa_operator.h"

using namespace libtosa;
// Demonstrate some basic assertions.
TEST(ParallelTests, BasicTest) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor a({10, 20, 30}, FLOAT, ws);
    Tensor b = Tensor::like(a);
    Tensor c = Tensor::like(a);

    parallel_for(Range(10), 
// {a,b,c}, input tensors
// 1. need to define the relation between the code and the input tensors
// 2. need to define the relation betwwen the code and the given index space
// 3. need to define index space mapping to tensor space
"\
    \
    c[i]=a[i]+b[i];\
    ");

}
