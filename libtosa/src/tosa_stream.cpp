//
// Created by rcohen on 08/06/2021.
//
#include <iostream>
#include <mutex>
#include <vector>
#include "tosa_stream.h"
#include "tosa_tensor.h"
#include "tosa_errors.h"
#include "tosa_operator.h"

using namespace libtosa;

Stream::Stream(int id):_id(id)
{
   //std::cout << "Stream " << id << " created." << std::endl;
}

void Stream::push(const std::shared_ptr<Operator> &op)
{
    add_single_op(op);
    for (auto t: op->outputs())
    {
        std::shared_ptr<Signal> signal = std::make_shared<Signal>("signal");
        add_single_op(signal);
        t.set_signal(signal);
    }
}

const OperatorPtr& Stream::pop()
{
    // todo: implement me
}

class libtosa::StreamPool
{
    std::mutex _pool_mutex;
    std::vector<Stream *> _pool;
    int _next_id;
    public:
        StreamPool(int init_size)
        {
            for (unsigned i=0; i<init_size; i++)
            {
                _pool.push_back(new Stream(i));
            }
            _next_id = init_size;
        }

        StreamPtr createStream()
        {
            std::lock_guard<std::mutex> guard(_pool_mutex);
            if (is_empty())
            {
                return StreamPtr(new Stream(_next_id++), 
                    [=](Stream* stream) {  returnStream(stream);});
            }
            return StreamPtr(getStream(), 
                    [=](Stream* stream) {  returnStream(stream);});
        }

    private:
        void returnStream(Stream *str)
        {
            //std::cout << "returning stream" << str->_id << std::endl;
            _pool.push_back(str);
        }

        Stream *getStream()
        {
            auto ret = _pool.back();
            _pool.pop_back();
            return ret;
        }
        bool is_empty() { return _pool.empty();}
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

