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

using namespace libtosa;


class CPUStream: public Stream {
    friend class StreamPool;        
    std::queue<CommandPtr> _cmd_queue; 
    std::atomic<bool>      mRun; // Use a race condition safe data 
                                 // criterium to end that thread loop
    std::thread mThread;

public:       
    CPUStream(int id):
        Stream(id), mRun(true) 
    {
        mThread = std::thread(&CPUStream::execute_queue, this);
    }
    ~CPUStream()
    {
        mRun = false; // <<<< Signal the thread loop to stop
        mThread.join(); // <<<< Wait for that thread to end
    }

protected:
    void push_impl(const CommandPtr &cmd) { _cmd_queue.push(cmd);}                
private:
    void execute_queue()
    {
        while (mRun == true)
        {
            while (!_cmd_queue.empty())
            {
                auto cmd = _cmd_queue.front();
                _cmd_queue.pop();
                auto cpu_cmd = dynamic_cast<CPUCommand*>(cmd.get());
                cpu_cmd->execute();
            }
        }
        back_to_idle();
    }
};

class libtosa::StreamPool
{
    std::mutex _pool_mutex;
    std::vector<Stream *> _ready_pool;
    int _next_id;
    public:
        StreamPool(int init_size)
        {
            for (unsigned i=0; i<init_size; i++)
            {
                _ready_pool.push_back(new CPUStream(i));
            }
            _next_id = init_size;
        }

        StreamPtr createStream()
        {
            std::lock_guard<std::mutex> guard(_pool_mutex);
            if (is_empty())
            {
                return StreamPtr(new CPUStream(_next_id++), 
                    [=](Stream* stream) {  returnStream(stream);});
            }
            return StreamPtr(getStream(), 
                    [=](Stream* stream) {  returnStream(stream);});
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

