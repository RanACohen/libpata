#include <iostream>
#include <gtest/gtest.h>

#include "tosa_tensor.h"
#include "tosa_operator.h"
#include "tosa_stream.h"
#include "tosa_backend.h"

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

TEST(ParallelTests, CreateStreamsTest) {
    auto str = StreamManager::Inst().createStream();
    auto str2 = StreamManager::Inst().createStream();
    ASSERT_EQ(str2->id(), 3);
    str.reset();
    str2 = StreamManager::Inst().createStream();
    ASSERT_EQ(str2->id(), 4);
    StreamManager::Inst().wait_for_all();
}


TEST(ParallelTests, WaitAllTest) {
    auto ws = std::make_shared<Workspace>(1000000);        
    auto str = StreamManager::Inst().createStream();
    int v=0;
    auto cmd = BackendManager::Inst().backend()->createTestCmd(&v, 8, 30);
    str->push(cmd);
    StreamManager::Inst().wait_for_all();
    ASSERT_EQ(v, 8);
}

TEST(ParallelTests, PushStreamsTest) {
    auto ws = std::make_shared<Workspace>(1000000);        
    auto str = StreamManager::Inst().createStream();
    int v=0;
    auto cmd = BackendManager::Inst().backend()->createTestCmd(&v, 8, 30);
    auto sig = BackendManager::Inst().backend()->createSignal();
    str->push(cmd);
    str->push(sig);
    str->wait_for_idle();
    ASSERT_EQ(v, 8);
    StreamManager::Inst().wait_for_all();
}

// todo: add signal testing + view signal

// todo: add view overlap testing

