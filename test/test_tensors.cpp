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

    for (int i=0; i<10 ; i++)
        for (int j=0; j<20 ; j++)
            for (int k=0; k<20 ; k++)
            {
                *t.at<float>(i,j,k) = i*10000.+j*100.+k;
            }
    Tensor s1 = t[{Range(5, 15), Range(15), Range(10, 20, 2)}];
    Tensor s2 = t.subrange(Range(5, 15, 2));
    EXPECT_EQ(s1.stride(), Shape({600,30,2}));
    EXPECT_EQ(s1.shape(), Shape({5,15,5}));
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_FALSE(s1.is_contiguous());
    EXPECT_FALSE(s2.is_contiguous());

    float *pf = s1.at<float>(0,5,1);
    EXPECT_EQ(50512. , *pf);
    
    EXPECT_EQ(50514. , *s1.at<float>(0,5,2));

    EXPECT_EQ(s2.stride(), Shape({1200,30,1}));
    EXPECT_EQ(s2.shape(), Shape({3,20,30}));
    EXPECT_EQ(70501. , *s2.at<float>(1,5,1));
    EXPECT_EQ(90502. , *s2.at<float>(2,5,2));

}

TEST(UtilsTests, TestTensorOverlap) {
 auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({10, 20, 30}, FLOAT, ws);
    Tensor s1 = t.subrange(Range(1), Range(0, 10));
    Tensor s2 = t[{Range(1), Range(5, 15)}];
    
    s1.mark_not_ready();
    ASSERT_FALSE(s2.is_ready());
    ASSERT_FALSE(t.is_ready());
}

TEST(UtilsTests, TestNonTensorOverlap) {
 auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({10, 20, 30}, FLOAT, ws);
    Tensor s1 = t.subrange(Range(1), Range(0, 5));
    Tensor s2 = t[{Range(1), Range(5, 15)}];
    
    s1.mark_not_ready();
    ASSERT_TRUE(s2.is_ready());
    ASSERT_FALSE(t.is_ready());
}


TEST(UtilsTests, TestNonTensorOverlapInterleaved) {
 auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({10, 20, 30}, FLOAT, ws);
    Tensor s1 = t.subrange(Range(1), Range(0, 10, 2));
    Tensor s2 = t[{Range(1), Range(5, 15, 2)}];
    
    s1.mark_not_ready();
    ASSERT_TRUE(s2.is_ready());
    ASSERT_FALSE(t.is_ready());
}

TEST(UtilsTests, TestStopWatch) {
    StopWatch timer;
    timer.stop();
    timer.stop_time = 123;
    LOG() << timer << "\n";
    timer.stop_time = 1023;
    LOG() << timer << "\n";
    timer.stop_time = 6302301;
    LOG() << timer << "\n";
    timer.stop_time = 63003023;
    LOG() << timer << "\n";
    timer.stop_time = 3663123111;
    LOG() << timer << "\n";
    timer.stop_time = 243663123654;
    LOG() << timer << "\n";
}

TEST(UtilsTests, TestList) {
    SafeGrowOnlyList<int, 8> L(3,5,8,7);
    ASSERT_EQ(L.size, 4);
    ASSERT_EQ(L[2], 8);

    SafeGrowOnlyList<int, 8> L2=L;
    ASSERT_EQ(L2.size, 4);
    ASSERT_EQ(L2[2], 8);

    SafeGrowOnlyList<int, 8> L3(L);
    ASSERT_EQ(L3.size, 4);
    ASSERT_EQ(L3[2], 8);

    SafeGrowOnlyList<int, 8> L4;
    L4.add(3); L4.add(5); L4.add(8); L4.add(7);
    ASSERT_EQ(L4.size, 4);
    ASSERT_EQ(L4[2], 8);


}


TEST(TensorPerformanceTests, TestTimeMeasure) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({10, 20, 30}, FLOAT, ws);
    auto x = reluN(t);
    x.sync();

    for (auto& time : schedule_time_map)
        std::cout << "Operation took " << time.count() << "usec. \n";
}

extern std::atomic<size_t> deadlock_put_index;

TEST(TensorPerformanceTests, TestAdd1000) {
    deadlock_put_index=0;
    auto ws = std::make_shared<Workspace>(10000000000);
    Tensor t({512, 1024}, FLOAT, ws);    
    t.fill(0.f, 0.25f);
    Tensor s1 = t.subrange(Range(256));
    Tensor s2 = t[{Range(256,512)}];
    ASSERT_FLOAT_EQ(*s1.at<float>(1,1), (0.25+s1.shape()[1]*0.25));
    ASSERT_FLOAT_EQ(*s2.at<float>(1,1), (1+s2.shape()[1]*(1+s1.shape()[0]))*0.25);
    Tensor x = s1;
    EXPECT_EQ(s1.shape(), s2.shape());
    StopWatch timer;
    for (unsigned i=0; i<1000; i++)
    {
        //x = x+s2;
        Add(x, s2, x); // inplace does not work, I have a deadlock.
    }
    std::cout << "Scheudling took " << timer << "\n";
    ASSERT_FLOAT_EQ(*x.at<float>(1,1), 65792272.0f);
    std::cout << "Operation took " << timer << "\n";    
}


TEST(TensorPerformanceTests, TestOverhead) {
    deadlock_put_index=0;
    auto ws = std::make_shared<Workspace>(100000000);
    Tensor x({16, 8}, FLOAT, ws);    
    x.fill(0.f, 0.25f);    
    Tensor s2({16, 8}, FLOAT, ws);
    s2.fill(1.f, 0.25f);    
    StopWatch timer;
    const unsigned n = 10000;
    for (unsigned i=0; i<n; i++)
    {
        //x = x+s2;
        Add(x, s2, x);
    }
    std::cout << "Scheudling took " << timer << " and " << timer/n << " per iteration\n";
    x.sync();
    ASSERT_FLOAT_EQ(*x.at<float>(0,0), n*1.f);
    std::cout << "Operation took " << timer << "\n";    
}

