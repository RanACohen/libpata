#include <iostream>
#include <gtest/gtest.h>
#include <memory>
#include <cassert>

#include "pata_debug.h"
#include "pata_utils.h"
#include "pata_tensor.h"
#include "pata_operator.h"
#include "pata_backend.h"

using namespace libpata;
std::ostream &LOG();
TEST(TensorOperationTests, TestReluN) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({10, 20, 30}, FLOAT, ws);
    auto x = reluN(t);
    x.sync();    
}

TEST(TensorOperationTests, TestAbs) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({10, 20, 30}, FLOAT, ws);
    auto x = abs(t);
    x.sync();
}

TEST(TensorOperationTests, TestAdd1) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({20, 30}, FLOAT, ws);
    t.fill(0.f,0.1f);    
    Tensor s1 = t.subrange(Range(2), Range(0, 10));
    Tensor s2 = t[{Range(2), Range(5, 15)}];
    ASSERT_FLOAT_EQ(*s1.at<float>(1,1), 3.1f);
    ASSERT_FLOAT_EQ(*s2.at<float>(1,1), 3.6f);

    EXPECT_EQ(s1.shape(), s2.shape());
    auto x = s1 + s2;    
    //StreamManager::Inst().wait_for_all();
    ASSERT_FLOAT_EQ(*x.at<float>(1,1), 6.7f);
    BackendManager::Inst().backend()->wait_for_all();
}
#define COL_SIZE 16384
#define ROWS  1024


TEST(TensorOperationTests, TestSerialAdd2D) {
    auto ws = std::make_shared<Workspace>(COL_SIZE*ROWS*4*4);
    Tensor a({ROWS, COL_SIZE}, FLOAT, ws);
    a.fill(2.0f);
    Tensor b({ROWS, COL_SIZE}, FLOAT, ws);
    b.fill(2.0f);
    TensorsList out_tiles;
    StopWatch timer;
    auto out = a+b;    
    BackendManager::Inst().backend()->wait_for_all();
    std::cout << "Operation took " << timer << "\n";
    ASSERT_FLOAT_EQ(*out.at<float>(1,1), 4.0f);
}


TEST(TensorOperationTests, TestMatMul1Tile) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor a({2, 3}, FLOAT, ws);
    //a.fill(2.0f);
    *a.at<float>(0,0) = 0.f;
    *a.at<float>(0,1) = 0.75f;
    *a.at<float>(0,2) = 0.25f;
    *a.at<float>(1,0) = 1.f;
    *a.at<float>(1,1) = 0.5f;
    *a.at<float>(1,2) = 1.25f;
    
    Tensor b({3, 2}, FLOAT, ws);
    b.fill(1.f);

    Tensor out({2, 2}, FLOAT, ws);

    // libxsmm expects the output tensor to be initialized to zeros.
    // fill function expects the tensor to be continous (if out is a view - it will not be continous)
    out.fill(0.f);

    TensorsList out_tiles;
    MatMul(a, b, out, out_tiles);
    EXPECT_EQ(out_tiles.size(), 1);

    ASSERT_FLOAT_EQ(*out.at<float>(0,0), 0.75f);
    ASSERT_FLOAT_EQ(*out.at<float>(0,1), 3.f);
    ASSERT_FLOAT_EQ(*out.at<float>(1,0), 0.75f);
    ASSERT_FLOAT_EQ(*out.at<float>(1,1), 3.f);
}


TEST(TensorOperationTests, TestLibxsmm) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor a({20, 30}, FLOAT, ws);
    a.fill(2.0f);
    Tensor b({30, 20}, FLOAT, ws);
    b.fill(2.0f);
    Tensor out({20, 20}, FLOAT, ws);
    out.fill(0.f);

    EXPECT_TRUE(test_Libxsmm(a,b,out));

    ASSERT_FLOAT_EQ(*out.at<float>(1,1), 120.0);
    ASSERT_FLOAT_EQ(*out.at<float>(10,10), 120.0);
    ASSERT_FLOAT_EQ(*out.at<float>(2,7), 120.0f);    

}

