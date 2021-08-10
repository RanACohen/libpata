#include <iostream>
#include <gtest/gtest.h>

#include "pata_debug.h"
#include "pata_tensor.h"
#include "pata_operator.h"
#include "pata_backend.h"

using namespace libpata;
// Demonstrate some basic assertions.

TEST(ParallelTests, TestAsyncBasic) {
    auto ws = std::make_shared<Workspace>(1000000);        
    auto backend = BackendManager::Inst().backend();
    int v=0;

    auto cmd = backend->createTestCmd(&v, 8, 300);
    auto sig = std::make_shared<Signal>();
    cmd->add_out_signal(sig);
    auto barrier = backend->createBarrierCmd();
    barrier->wait_on_signal(sig); // the signal to wait for         
    LOG() << "Before schedule\n";
    backend->schedule(cmd);
    LOG() << "After schedule\n";
    ASSERT_EQ(v, 0);
    barrier->wait();
    LOG() << "After barrier sync\n";
    ASSERT_EQ(v, 8);
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
    out.fill(0.0f);
    timer.start();
    Add2D(a, b, out, out_tiles, ROWS/NTHREADS);
    ASSERT_EQ(out_tiles.size(), NTHREADS);
    ASSERT_FLOAT_EQ(*out.at<float>(1,1), 4.0f);
    timer.stop();
    out_tiles.clear();
    auto par_dur = timer.leap_usec();
    std::cout << "Parallel Operation took " << timer << "\n";
    out.fill(0.0f);
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
    std::cout << "Done add " << timer << "\n";   
    ASSERT_FLOAT_EQ(*x.at<float>(1,1), 6.0f);
    std::cout << "Test Operation took " << timer << "\n";   
}
