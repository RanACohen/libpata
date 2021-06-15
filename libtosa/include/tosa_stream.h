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

    class Stream {
        int _id;

    public:
        Stream(int id):_id(id){};
        
        int id() { return _id;}
        virtual void push(const CommandPtr &cmd) = 0; 
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
