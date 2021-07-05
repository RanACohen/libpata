//
// Created by rcohen on 15/6/2021.
//
#pragma once
#ifndef LIBPATA_PATA_COMMAND_H
#define LIBPATA_PATA_COMMAND_H
#include <memory>
#include <mutex>
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
        public:            
            Signal() = default;
            virtual ~Signal()=0; // mark trhis abstract, must inherit!

            virtual std::shared_ptr<Wait> getWaitCmd() = 0;
            virtual void mark_not_ready() =0; 
            virtual void mark_ready() = 0;
            virtual bool is_ready() = 0;
    };

    class Wait: public virtual Command {
        public:
            Wait(const std::shared_ptr<Signal>& wait_on):_wait_on(wait_on){}
            virtual ~Wait()=0; // abstract, must inherit
        protected:
            std::shared_ptr<Signal> _wait_on;

    };
};

#endif //LIBPATA_PATA_COMMAND_H