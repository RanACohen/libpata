//
// Created by rcohen on 15/6/2021.
//
#pragma once
#ifndef LIBXLA_XLA_COMMAND_H
#define LIBXLA_XLA_COMMAND_H
#include <memory>
#include <mutex>
#include <condition_variable>

namespace libxla {    
    class Command: public std::enable_shared_from_this<Command> {
        public:
        virtual ~Command() = default;
    };
    typedef std::shared_ptr<Command> CommandPtr;


    class Wait;
    class Signal : public virtual Command {
        private:            
            bool _ready = false;
        public:            
            Signal(){} 
            virtual ~Signal()=0; // mark trhis abstract, must inherit!

            virtual std::shared_ptr<Wait> getWaitCmd() = 0;

            inline void signal() { _ready = true;}
            inline bool is_ready() { return _ready; }
    };

    class Wait: public virtual Command {
        public:
            Wait(const std::shared_ptr<Signal>& wait_on):_wait_on(wait_on){}
            virtual ~Wait()=0; // abstract, must inherit
        protected:
            std::shared_ptr<Signal> _wait_on;

    };
};

#endif //LIBXLA_XLA_COMMAND_H