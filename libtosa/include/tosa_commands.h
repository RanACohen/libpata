//
// Created by galstar on 31/5/2021.
//
#pragma once
#ifndef LIBTOSA_TOSA_COMMAND_H
#define LIBTOSA_TOSA_COMMAND_H
#include <string>
#include <memory>

using namespace std;

namespace libtosa {    
    class Command {        
    };
    typedef std::shared_ptr<Command> CommandPtr;


    class Signal : public Command {
        private:            
            bool _ready = false;
        public:            
            Signal(){}            
            inline void signal() { _ready = true;}
            inline bool is_ready() { return _ready; }
    };

    class Wait: public Command {
        std::shared_ptr<Signal> _wait_on;
        public:
            Wait(const std::shared_ptr<Signal>& wait_on):_wait_on(wait_on){}
    };

};

#endif //LIBTOSA_TOSA_COMMAND_H