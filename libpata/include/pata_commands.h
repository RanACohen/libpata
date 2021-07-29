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
    class Wait;
    class Command;

    class Signal {
        protected:
            bool _ready = false;
            std::vector<std::shared_ptr<Wait>> _waiting_on_me;
        public:            
            Signal() = default;
            virtual ~Signal()=0; // mark trhis abstract, must inherit!
            
            inline void signal() { _ready = true;}
            inline bool is_ready() { return _ready; }
            inline void add_wait(const std::shared_ptr<Wait> &wait) {
                _waiting_on_me.push_back(wait);
            }

            Command *_dbg_cmd_src;

    };
    typedef std::shared_ptr<Signal> SignalPtr;
    typedef std::vector<SignalPtr> SignalList;

    class Command: public std::enable_shared_from_this<Command> {
        public:
            Command(const std::string &name):_cmd_name(name){
                static std::atomic<unsigned> cmd_id_gen(0);
                _cmd_id = cmd_id_gen++;
            }
            virtual ~Command() = default;
            inline unsigned id() const { return _cmd_id;}

            void add_signal(const SignalPtr &signal) { 
                _out_signals.push_back(signal);
                signal->_dbg_cmd_src = this;
            }
            inline const SignalList &get_signals() { return _out_signals; }
        
            Wait *_dbg_waiting_on;
        protected:
            std::string _cmd_name;
            unsigned _cmd_id;
            SignalList _out_signals; // signals to mark when done processing this command
    };
    typedef std::shared_ptr<Command> CommandPtr;

    class Wait: public std::enable_shared_from_this<Wait> {
        public:
            Wait(const CommandPtr&cmd):_cmd_waiting(cmd){
                if (cmd) cmd->_dbg_waiting_on = this;
            };
            virtual ~Wait() = default;
            void wait_on_signal(const SignalPtr &signal) { 
                _wait_on.push_back(signal);
                signal->add_wait(shared_from_this());
            }

            
            bool is_ready() const { 
                for (auto &sig: _wait_on)
                {
                    auto s = sig.lock();
                    if (s && !s->is_ready()) // if signal object lost, it means the tensor is lost and thus it must be ready
                        return false;
                }
                return true;
            }
            CommandPtr cmd_waiting() const { return _cmd_waiting;}
        protected:
            std::vector<std::weak_ptr<Signal>> _wait_on;        // siganls I am waiting to be ready
            CommandPtr _cmd_waiting;
    };

    class Barrier: public Wait {
    public:
        Barrier():Wait(nullptr){}
        virtual void wait() = 0;
    };
    
};

#endif //LIBPATA_PATA_COMMAND_H