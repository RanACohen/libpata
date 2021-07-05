#include <iostream>
#include <gtest/gtest.h>
#include <memory>
#include <cassert>

#include "pata_utils.h"
#include "pata_tensor.h"
#include "pata_operator.h"
#include "pata_backend.h"

using namespace libpata;

TEST(TensorOperationTests, TestReluN) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({10, 20, 30}, FLOAT, ws);
    auto x = reluN(t);
    BackendManager::Inst().backend()->wait_for_all();
}

TEST(TensorOperationTests, TestAbs) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({10, 20, 30}, FLOAT, ws);
    auto x = abs(t);
    BackendManager::Inst().backend()->wait_for_all();
}

TEST(TensorOperationTests, TestAdd1) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({20, 30}, FLOAT, ws);
    float *ptF = (float*)t.base_addr();
    for (unsigned i=0; i<t.volume(); i++) ptF[i]=i*0.1;
    Tensor s1 = t.subrange(Range(2), Range(0, 10));
    Tensor s2 = t[{Range(2), Range(5, 15)}];
    ASSERT_EQ(*s1.at<float>(1,1), 3.1f);
    ASSERT_EQ(*s2.at<float>(1,1), 3.6f);

    EXPECT_EQ(s1.shape(), s2.shape());
    auto x = s1 + s2;    
    //StreamManager::Inst().wait_for_all();
    ASSERT_FLOAT_EQ(*x.at<float>(1,1), 6.7f);
    BackendManager::Inst().backend()->wait_for_all();
}

TEST(TensorOperationTests, TestParallelAdd2D) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor a({20, 30}, FLOAT, ws);
    a.fill(2.0f);
    Tensor b({20, 30}, FLOAT, ws);
    b.fill(2.0f);
    Tensor out({20, 30}, FLOAT, ws);
    out.fill(0.f);
    TensorsList out_tiles;
    
    //Add2D(a, b, out, out_tiles);
    //EXPECT_EQ(out_tiles.size(), 2);

    //BackendManager::Inst().backend()->wait_for_all();

    //ASSERT_FLOAT_EQ(*out.at<float>(1,1), 4.0f);
}

TEST(TensorOperationTests, TestMatMul1Tile) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor a({20, 30}, FLOAT, ws);
    a.fill(0.0f, 0.25f);
    Tensor b({30, 20}, FLOAT, ws);
    b.fill(-36.0f, 0.125f);
    Tensor out_base({30, 30}, FLOAT, ws);
    Tensor out = out_base[{Range(20), Range(20)}];

    TensorsList out_tiles;
    MatMul(a, b, out, out_tiles);
    EXPECT_EQ(out_tiles.size(), 1);

    ASSERT_FLOAT_EQ(*out.at<float>(1,1), 1529.84375f);
    ASSERT_FLOAT_EQ(*out.at<float>(10,10), 4942.8125f);
    ASSERT_FLOAT_EQ(*out.at<float>(2,7), 2033.28125f);    
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

