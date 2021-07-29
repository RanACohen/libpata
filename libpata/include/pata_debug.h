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

std::ostream &LOG();

class StopWatch
{
    bool is_running = true;
    std::chrono::high_resolution_clock::time_point start_time = std::chrono::high_resolution_clock::now();
public:
    unsigned long stop_time=0;
    inline void start() {start_time = std::chrono::high_resolution_clock::now(); is_running = true;}
    inline void stop() { if (is_running) stop_time = leap_usec(); is_running = false;}
    inline unsigned long leap_usec() const { 
        return is_running ? std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now()-start_time).count() : stop_time;
    }
    friend std::ostream& operator<<(std::ostream& os, const StopWatch& watch);
};
std::ostream& operator<<(std::ostream& os, const StopWatch& watch);

#endif // LIBXLA_XLA_DEBUG_H