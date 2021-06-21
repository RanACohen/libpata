//
// Created by rcohen on 08/06/2021.
//
#pragma once
#ifndef LIBTOSA_TOSA_STREAM_HPP
#define LIBTOSA_TOSA_STREAM_HPP
#include <memory>
#include <queue>
#include <iostream>

#include "tosa_commands.h"

namespace libtosa {  
    
    class Stream: public std::enable_shared_from_this<Stream> {
        int _id;
        std::shared_ptr<Stream> _myself;
        std::condition_variable _cv;
        std::mutex _wait_mutex;
        bool _idle;

    public:
        Stream(int id):_id(id),_idle(true){};
        virtual ~Stream() = default;
        
        int id() { return _id;}
        void push(const CommandPtr &cmd)
        {
            _idle = false;
            //std::cout << "Stream " << id() << " BUSY \n";
            _myself = shared_from_this();
            push_impl(cmd);
        }
        void back_to_idle()
        { 
            //std::cout << "Stream " << id() << " IDLE \n";
            _myself.reset(); 
            _idle = true;
            _cv.notify_all();
        } 
        void wait_for_idle(){
            //std::cout << "Stream " << id() << " wait for IDLE is " << (_idle ? "IDLE":"BUSY") << "\n";
            std::unique_lock<std::mutex> lk(_wait_mutex);
            _cv.wait(lk, [=]{return _idle;});
            //std::cout << "Stream " << id() << " WAIT DONE \n";
        };

    protected:
        virtual void push_impl(const CommandPtr &cmd) = 0;
    };
    
    typedef std::shared_ptr<Stream> StreamPtr;
    typedef Stream*(*StreamCreatorFunc)(int);
};

#endif //LIBTOSA_TOSA_STREAM_HPP
