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

void log_dead_lock(int wait_id, int sig_id, int sig_str_id, EventType event);

#endif // LIBXLA_XLA_DEBUG_H