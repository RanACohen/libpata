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
ScheduleTimeMeasurement libpata::schedule_time_map;

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

TEST(UtilsTests, TestListRef) {
    auto ws = std::make_shared<Workspace>(1000000);
    std::list<std::weak_ptr<TensorImpl>> a_list;
    std::list<WeakListReference<TensorImpl>> refernces;
    std::shared_ptr<TensorImpl> t(new TensorImpl({10, 20, 30}, FLOAT, ws));
    refernces.emplace_back(&a_list, t);
    refernces.clear();
    EXPECT_TRUE(a_list.empty());    
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
    timer.stop_time = 1123;
    LOG() << timer << "\n";
    timer.stop_time = 63123;
    LOG() << timer << "\n";
    timer.stop_time = 63123123;
    LOG() << timer << "\n";
    timer.stop_time = 3663123111;
    LOG() << timer << "\n";
    timer.stop_time = 243663123654;
    LOG() << timer << "\n";

}

TEST(TensorPerformanceTests, TestTimeMeasure) {
    auto ws = std::make_shared<Workspace>(1000000);
    Tensor t({10, 20, 30}, FLOAT, ws);
    auto x = reluN(t);
    BackendManager::Inst().backend()->wait_for_all();

    for (auto& t : schedule_time_map)
        std::cout << "Operation took " << t.count() << "usec. \n";
}

extern std::atomic<size_t> deadlock_put_index;

TEST(TensorPerformanceTests, TestAdd1000) {
    deadlock_put_index=0;
    auto ws = std::make_shared<Workspace>(10000000000);
    Tensor t({512, 1024}, FLOAT, ws);
    float *ptF = (float*)t.base_addr();
    for (unsigned i=0; i<t.volume(); i++) ptF[i]=i*0.25;
    Tensor s1 = t.subrange(Range(256));
    Tensor s2 = t[{Range(256,512)}];
    ASSERT_FLOAT_EQ(*s1.at<float>(1,1), (0.25+s1.shape()[1]*0.25));
    ASSERT_FLOAT_EQ(*s2.at<float>(1,1), (1+s2.shape()[1]*(1+s1.shape()[0]))*0.25);
    Tensor x = s1;
    EXPECT_EQ(s1.shape(), s2.shape());
    StopWatch timer;
    for (unsigned i=0; i<100; i++)
    {
        x = x + s2;
    }
    ASSERT_FLOAT_EQ(*x.at<float>(1,1), 6579472.0f);
    std::cout << "Operation took " << timer << "\n";
    BackendManager::Inst().backend()->wait_for_all();
}

