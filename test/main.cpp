#include <iostream>
#include <gtest/gtest.h>

#include "tosa_tensor.h"

using namespace libtosa;
// Demonstrate some basic assertions.
TEST(TensorTests, StrideTest) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({10, 20, 30}, FLOAT, ws);
    Shape ex_stride = {600,30,1};
    EXPECT_EQ(t.stride(), ex_stride);
}

TEST(TensorTests, TestView1) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({10, 20, 30}, FLOAT, ws);
    Tensor s1(t, {Range(5, 15), Range(15), Range(10, 20, 2)});
    Tensor s2(t, {Range(5, 15, 2)});
    EXPECT_EQ(s1.stride(), Shape({600,30,2}));
    EXPECT_EQ(s1.shape(), Shape({5,15,5}));

    EXPECT_EQ(s2.stride(), Shape({1200,30,1}));
    EXPECT_EQ(s2.shape(), Shape({3,20,30}));

}

