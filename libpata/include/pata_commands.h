//
// Created by rcohen on 15/6/2021.
//
#pragma once
#ifndef LIBPATA_PATA_COMMAND_H
#define LIBPATA_PATA_COMMAND_H
#include <memory>
#include <mutex>
#include <list>
#include <vector>
#include <condition_variable>
#include <atomic>

namespace libpata {        
    class Command;
    typedef std::shared_ptr<Command> CommandPtr;
    typedef std::vector<CommandPtr> CommandList;

    class Signal {
        protected:
            bool _ready = false;
            CommandList _waiting_on_me;
            std::mutex _wait_list_guard;
        public:            
            Signal() = default;
            //virtual ~Signal()=0; // mark trhis abstract, must inherit!
                        
            inline bool is_ready() const {                 
                return _ready; 
            }

            inline void mark_ready() {
                _ready = true;
            }

            inline void add_wait(const CommandPtr &waited_cmd) {
                std::unique_lock<std::mutex> lk(_wait_list_guard);
                _waiting_on_me.push_back(waited_cmd);
            }

            inline CommandList &get_waited_commands() { return _waiting_on_me; }

            std::weak_ptr<Command> _signals_me; // for debug

    };
    typedef std::shared_ptr<Signal> SignalPtr;
    typedef std::vector<SignalPtr> SignalList;
    typedef std::vector<std::weak_ptr<Signal>> WeakSignalList;

    class Command: public std::enable_shared_from_this<Command> {
        public:
            Command(const std::string &name):_cmd_name(name){
                static std::atomic<unsigned> cmd_id_gen(0);
                _cmd_id = cmd_id_gen++;
            }
            virtual ~Command() = default;
            inline unsigned id() const { return _cmd_id;}

            void add_out_signal(const SignalPtr &signal) { 
                _out_signals.push_back(signal);
                signal->_signals_me = shared_from_this();
            }
            void wait_on_signal(const SignalPtr &signal) {
                std::unique_lock<std::mutex> lk(_wait_list_mx);
                _wait_on_signals.push_back(signal);
                signal->add_wait(shared_from_this());
            }

            //inline const SignalList &get_out_signals() { return _out_signals; }        
        
            bool is_ready() { 
                std::unique_lock<std::mutex> lk(_wait_list_mx);
                for (auto &sig: _wait_on_signals)
                {
                    auto s = sig.lock();
                    if (s && !s->is_ready()) // if signal object lost, it means the tensor is lost and thus it must be ready
                        return false;
                }
                return true;
            }
            
        protected:
            std::string _cmd_name;
            unsigned _cmd_id;
            std::mutex _wait_list_mx; // guards the _wait_on list
            WeakSignalList _wait_on_signals; // signals that needs to be ready for executing this command
            SignalList _out_signals; // signals to mark when done processing this command
    };
    


    class Barrier: public Command {
    public:
        Barrier():Command("barrier"){}
        virtual void wait() = 0;
    };
    
};

#endif //LIBPATA_PATA_COMMAND_H