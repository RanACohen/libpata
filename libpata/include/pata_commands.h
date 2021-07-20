//
// Created by rcohen on 15/6/2021.
//
#pragma once
#ifndef LIBPATA_PATA_COMMAND_H
#define LIBPATA_PATA_COMMAND_H
#include <memory>
#include <mutex>
#include <vector>
#include <condition_variable>
#include <atomic>

namespace libpata {    
    class Stream;
    class Command: public std::enable_shared_from_this<Command> {
        public:
            Stream *sched_in_stream;
            virtual ~Command() = default;
    };
    typedef std::shared_ptr<Command> CommandPtr;


    class Wait;
    class Signal : public virtual Command {
        protected:
            bool _ready = false;
 
        public:            
            Signal() = default;
            virtual ~Signal()=0; // mark trhis abstract, must inherit!
            
            inline void signal() { _ready = true;}
            inline bool is_ready() { return _ready; }
    };
    typedef std::shared_ptr<Signal> SignalPtr;
    typedef std::vector<SignalPtr> SignalList;

    class Wait: public virtual Command {
        public:
            Wait() = default;
            void add_signal(const SignalPtr &signal) { _wait_on.push_back(signal);}
            virtual ~Wait()=0; // abstract, must inherit
            bool is_empty() const { return _wait_on.empty(); }
        protected:
            SignalList _wait_on;
    };
};

#endif //LIBPATA_PATA_COMMAND_H