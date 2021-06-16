//
// Created by rcohen on 08/06/2021.
//
#pragma once
#ifndef LIBTOSA_TOSA_STREAM_HPP
#define LIBTOSA_TOSA_STREAM_HPP
#include <memory>
#include <queue>

#include "tosa_commands.h"
namespace libtosa {  
    class StreamPool;

    class Stream: public std::enable_shared_from_this<Stream> {
        int _id;
        std::shared_ptr<Stream> _myself;

    public:
        Stream(int id):_id(id){};
        
        int id() { return _id;}
        void push(const CommandPtr &cmd)
        {
            _myself = shared_from_this();
            push_impl(cmd);
        }
        void back_to_idle() // todo: call this from backend
        { 
            //todo: add mutex
            _myself.reset(); 
        } 
    protected:
        virtual void push_impl(const CommandPtr &cmd) = 0;
    };

    typedef std::shared_ptr<Stream> StreamPtr;

    class StreamManager {
        public:
            static StreamManager &Inst();
            StreamPtr createStream(); 

    private:
        std::shared_ptr<StreamPool> _pool;
        StreamManager();
    };
};

#endif //LIBTOSA_TOSA_STREAM_HPP
