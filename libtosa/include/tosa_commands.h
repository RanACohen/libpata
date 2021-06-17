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
    class Command {
        public:
        virtual ~Command() = default;
    };
    typedef std::shared_ptr<Command> CommandPtr;

    class CPUCommand: public Command
    {
        public:
            virtual ~CPUCommand() = default;
            virtual void execute() = 0;
    };

    class Signal : public Command {
        private:            
            bool _ready = false;
        public:            
            Signal(){}            
            inline void signal() { _ready = true;}
            inline bool is_ready() { return _ready; }
    };

    class CPUSignal: virtual public Signal, CPUCommand
    {
        std::condition_variable _cv;
        std::mutex _mutex;
        public:
            void wait() {
                 std::unique_lock<std::mutex> lk(_mutex);
                _cv.wait(lk, [=]{return is_ready();});
            }
            virtual void execute() {
                signal();
                _cv.notify_all();
            }
    };

    class Wait: public Command {
        public:
            Wait(const std::shared_ptr<Signal>& wait_on):_wait_on(wait_on){}
        
        protected:
            std::shared_ptr<Signal> _wait_on;

    };

    class CPUWait: virtual public Wait, CPUCommand
    {
        public:
         virtual void execute() {
             Signal * ps = _wait_on.get();
             auto sig = dynamic_cast<CPUSignal*>(ps);
            sig->wait();
         }
    };

};

#endif //LIBTOSA_TOSA_COMMAND_H