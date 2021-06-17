//
// Created by galstar on 31/5/2021.
//
#pragma once
#ifndef LIBTOSA_TOSA_COMMAND_H
#define LIBTOSA_TOSA_COMMAND_H
#include <string>
#include <memory>
#include <condition_variable>

using namespace std;

namespace libtosa {    
    class Command: public std::enable_shared_from_this<Command> {
        public:
        virtual ~Command() = default;
    };
    typedef std::shared_ptr<Command> CommandPtr;

    class CPUCommand: public virtual Command
    {
        public:
            virtual ~CPUCommand() = default;
            virtual void execute() = 0;
    };

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

    class CPUSignal: virtual public Signal, CPUCommand
    {
        std::condition_variable _cv;
        std::mutex _mutex;
        public:
            void wait();
            virtual void execute();
            virtual std::shared_ptr<Wait> getWaitCmd();
    };

    class CPUWait: virtual public Wait, CPUCommand
    {
        public:
        CPUWait(const std::shared_ptr<Signal>& wait_on): Wait(wait_on){}
        virtual void execute();
    };

};

#endif //LIBTOSA_TOSA_COMMAND_H