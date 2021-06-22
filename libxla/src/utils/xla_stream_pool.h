//
// Created by galstar on 21/06/2021
//
#pragma once
#ifndef LIBXLA_STREAM_POOL_HPP
#define LIBXLA_STREAM_POOL_HPP

#include <iostream>
#include <mutex>
#include <vector>
#include <thread>
#include "xla_stream.h"
#include "xla_backend.h"

namespace libxla {  
    class StreamPool
    {
        std::mutex _pool_mutex;
        std::vector<Stream *> _all;
        std::vector<Stream *> _ready_pool;
        StreamCreatorFunc _stream_obj_creator;
    public:
        inline StreamPool(int init_size, StreamCreatorFunc creator)
        {
            _stream_obj_creator = creator;
            for (unsigned i=0; i<init_size; i++)
            {
                auto str = (_stream_obj_creator)(i);
                _ready_pool.push_back(str);
                _all.push_back(str);
            }            
        }

        inline ~StreamPool()
        {
            std::cout << "Pool is shutting down...\n";
            wait_for_all();
            if (_ready_pool.size() != _all.size())
            {
                std::cout << "warning! stream is used by someone but system is shutting down...\n";
            }
            for (auto &str : _all)
            {
                std::cout << "deleting " << str->id() << " ...\n";
                delete str;
            }
        }
        
        inline StreamPtr createStream()
        {
            std::lock_guard<std::mutex> guard(_pool_mutex);
            Stream *ret_str = nullptr;
            if (is_empty())
            {
                ret_str = _stream_obj_creator(_all.size());                
                _all.push_back(ret_str);
            } else {
                ret_str = _ready_pool.back();
                _ready_pool.pop_back();
            }
            return StreamPtr(ret_str, 
                    [=](Stream* stream) {  returnStream(stream);});
        }
        inline void wait_for_all()
        {
            for (auto &str : _all)
                str->wait_for_idle();
        }
    private:
        inline void returnStream(Stream *str)
        {
            std::lock_guard<std::mutex> guard(_pool_mutex);
            //std::cout << "returning stream #" << str->id() << std::endl;
            _ready_pool.push_back(str);
        }
        
        inline bool    is_empty() { return _ready_pool.empty();}
    };
}

#endif //LIBXLA_STREAM_POOL_HPP
