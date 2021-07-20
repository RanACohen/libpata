#include <iostream>
#include <gtest/gtest.h>

#include "pata_debug.h"
#include "pata_tensor.h"
#include "pata_operator.h"
#include "pata_stream.h"
#include "pata_backend.h"

using namespace libpata;
// Demonstrate some basic assertions.
/*
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
    "c[i]=a[i]+b[i];\
    ");
}
*/
/* Ran: this test is invalid, we cannot anticipate pool id ordering...
need to think how e test the pool logic, maybe expose no of free streams?
TEST(ParallelTests, CreateStreamsTest) {
    auto backend = BackendManager::Inst().backend();
    backend->wait_for_all();
    auto str = backend->createStream();
    auto str2 = backend->createStream();
    

    str.reset();
    str2 = backend->createStream();
    ASSERT_EQ(str2->id(), 4);

    backend->wait_for_all();
}
*/

TEST(ParallelTests, WaitAllTest) {
    auto ws = std::make_shared<Workspace>(1000000);        
    auto backend = BackendManager::Inst().backend();
    int v=0;

    auto str = backend->createStream();
    auto cmd = backend->createTestCmd(&v, 8, 30);
    
    str->push(cmd);
    backend->wait_for_all();

    ASSERT_EQ(v, 8);
}

TEST(ParallelTests, PushStreamsTest) {
    auto backend = BackendManager::Inst().backend();
    auto ws = std::make_shared<Workspace>(1000000);        
    auto str = backend->createStream();
    int v=0;

    auto cmd = backend->createTestCmd(&v, 8, 30);
    auto sig = backend->createSignal();
    str->push(cmd);
    str->push(sig);
    str->wait_for_idle();
    
    ASSERT_EQ(v, 8);
    backend->wait_for_all();
}

// todo: add signal testing + view signal
#define COLS 4096
#define ROWS  4096
#define NTHREADS 8

TEST(ParallelTests, TestParallelAdd2DBoost) {
    
    auto ws = std::make_shared<Workspace>(COLS*ROWS*4*4);
    Tensor a({ROWS, COLS}, FLOAT, ws);
    a.fill(2.0f);
    Tensor b({ROWS, COLS}, FLOAT, ws);
    b.fill(2.0f);
    Tensor out({ROWS, COLS}, FLOAT, ws);
    out.fill(0.0f); // This is needed to allocate the output memory and warm the caches
    TensorsList out_tiles;
    //warm up
    StopWatch timer;
    timer.start();
    Add2D(a, b, out, out_tiles, ROWS/NTHREADS);
    ASSERT_EQ(out_tiles.size(), NTHREADS);
    ASSERT_FLOAT_EQ(*out.at<float>(1,1), 4.0f);
    timer.stop();
    out_tiles.clear();
    std::cout << "Parallel Operation took " << timer << "\n";
    timer.start();
    Add2D(a, b, out, out_tiles, ROWS/NTHREADS);
    ASSERT_EQ(out_tiles.size(), NTHREADS);
    ASSERT_FLOAT_EQ(*out.at<float>(1,1), 4.0f);
    timer.stop();
    auto par_dur = timer.leap_usec();
    std::cout << "Parallel Operation took " << timer << "\n";
    timer.start();
    Add2D(a, b, out, out_tiles, ROWS);
    ASSERT_FLOAT_EQ(*out.at<float>(1,1), 4.0f);
    timer.stop();
    auto ser_dur = timer.leap_usec();
    std::cout << "Serial Operation took " << timer << "\n";
    std::cout << "Parallel boosting factor " << (float)ser_dur/par_dur << "\n";
    BackendManager::Inst().backend()->wait_for_all();
}

TEST(ParallelTests, TestParallelAdd2DSync) {
    
    auto ws = std::make_shared<Workspace>(COLS*ROWS*4*4);
    Tensor a({ROWS, COLS}, FLOAT, ws);
    a.fill(2.0f);
    Tensor b({ROWS, COLS}, FLOAT, ws);
    b.fill(2.0f);
    Tensor out({ROWS, COLS}, FLOAT, ws);
    out.fill(0.0f); // This is needed to allocate the output memory and warm the caches
    TensorsList out_tiles;
    //warm up
    StopWatch timer;
    timer.start();
    Add2D(a, b, out, out_tiles, ROWS/NTHREADS);
    ASSERT_EQ(out_tiles.size(), NTHREADS);
    auto x = a+out;
    ASSERT_FLOAT_EQ(*x.at<float>(1,1), 6.0f);
    std::cout << "Test Operation took " << timer << "\n";
    BackendManager::Inst().backend()->wait_for_all();
}
