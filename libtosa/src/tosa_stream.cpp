//
// Created by rcohen on 08/06/2021.
//
#include <iostream>
#include <mutex>
#include <vector>
#include <thread>
#include <atomic>
#include "tosa_stream.h"
#include "tosa_tensor.h"
#include "tosa_errors.h"
#include "tosa_operator.h"
#include "tosa_backend.h"

using namespace libtosa;

class libtosa::StreamPool
{
    std::mutex _pool_mutex;
    std::vector<Stream *> _all;
    std::vector<Stream *> _ready_pool;    
    public:
        StreamPool(int init_size)
        {
            for (unsigned i=0; i<init_size; i++)
            {
                auto str = BackendManager::Inst().backend()->createStream(i);
                _ready_pool.push_back(str);
                _all.push_back(str);
            }            
        }

        ~StreamPool()
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

        StreamPtr createStream()
        {
            std::lock_guard<std::mutex> guard(_pool_mutex);
            if (is_empty())
            {
                auto new_stream = BackendManager::Inst().backend()->createStream(_all.size());
                _ready_pool.push_back(new_stream);
                _all.push_back(new_stream);
            }
            return StreamPtr(getStream(), 
                    [=](Stream* stream) {  returnStream(stream);});
        }

        void wait_for_all()
        {
            for (auto &str : _all)
            {
                str->wait_for_idle();
            }
        }

    private:
        void returnStream(Stream *str)
        {
            //std::cout << "returning stream" << str->_id << std::endl;
            _ready_pool.push_back(str);
        }

        Stream *getStream()
        {
            auto ret = _ready_pool.back();
            _ready_pool.pop_back();
            return ret;
        }
        bool is_empty() { return _ready_pool.empty();}
};


StreamManager &StreamManager::Inst()
{
    static StreamManager inst;

    return inst;
}

StreamManager::StreamManager()
{
    _pool = std::make_shared<StreamPool>(5);
}

StreamPtr StreamManager::createStream()
{
   return _pool->createStream();
}

void StreamManager::wait_for_all()
{
    _pool->wait_for_all();
}

