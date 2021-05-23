#include <iostream>
#include <gtest/gtest.h>

#include "tosa_tensor.h"

using namespace libtosa;
// Demonstrate some basic assertions.
TEST(TensorTests, StrideTest) {
    Tensor t({10,20,30}, FLOAT);
    Shape ex_stride = {600,30,1};
    EXPECT_EQ(t.stride(), ex_stride);
}