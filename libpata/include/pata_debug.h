//
// Created by rcohen on 23/6/2021.
//
#pragma once
#ifndef LIBXLA_XLA_DEBUG_H
#define LIBXLA_XLA_DEBUG_H
#include <atomic>
#include <chrono>
#include <iostream>

typedef enum {
    STREAM_WAIT,
    WAIT_WOKE_UP,
    SIGNAL_ON,
    CMD_PUSH,
    BACK_TO_IDLE,
    CMD_POP
} EventType;

void deadlock_debug_reset();

void dump_dead_lock();

#ifdef ENABLE_EVENT_TRACE
void log_dead_lock(int wait_id, int sig_id, int sig_str_id, EventType event);
#else
inline void log_dead_lock(int wait_id, int sig_id, int sig_str_id, EventType event) {};
#endif

void set_local_thread_id(unsigned id);

class FlushLog {};

std::ostream &LOG();

class StopWatch
{
    bool is_running = true;
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
public:
    unsigned long stop_time=0;
    inline void start() {start_time = std::chrono::high_resolution_clock::now(); is_running = true;}
    inline void stop() { if (is_running) stop_time = leap_nsec(); is_running = false;}
    inline unsigned long leap_nsec() const { 
        return is_running ? std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-start_time).count() : stop_time;
    }

    StopWatch operator/(unsigned num)
    {
        StopWatch ret;
        ret.stop();
        if (is_running) stop_time = leap_nsec();
        ret.stop_time = stop_time / num;
        return ret;
    }

    friend std::ostream& operator<<(std::ostream& os, const StopWatch& watch);    
};
std::ostream& operator<<(std::ostream& os, const StopWatch& watch);
std::ostream& operator<<(std::ostream& os, const FlushLog& );

#endif // LIBXLA_XLA_DEBUG_H