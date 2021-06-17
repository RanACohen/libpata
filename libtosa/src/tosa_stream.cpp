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

template<class T>
struct MutexFreeQueue {
    int _size;
    int _get;
    int _put;
    T* _cyclic_buffer;

    MutexFreeQueue(int size) {
        _size = size;
        _cyclic_buffer = new T[size];
        _get = 0;
        _put = 0;
    }
    ~MutexFreeQueue() {
        delete[] _cyclic_buffer;
    }
    void push(const T&item)
    {
        auto next_put = (_put+1)%_size;
        TOSA_ASSERT (next_put!=_get); // queue if full, nothing to do, bail out....
        _cyclic_buffer[_put] = item;
        _put = next_put;        
    }
    bool pop(T &ret)
    {
        if (empty()) return false;
        ret = _cyclic_buffer[_get];
        _get = (_get+1) % _size;
        return true;
    }

    inline bool empty() const { return _get == _put; }
};


class CPUStream: public Stream {
    friend class StreamPool;
    MutexFreeQueue<CommandPtr> _cmd_queue; 
    bool      mRun; // Use a race condition safe data 
                                 // criterium to end that thread loop
    std::thread mThread;
    std::condition_variable _cv;
    std::mutex _mutex;

public:       
    CPUStream(int id):
        Stream(id), mRun(true), _cmd_queue(1024)
    {
        mThread = std::thread(&CPUStream::execute_queue, this);
    }
    ~CPUStream()
    {
        mRun = false; // <<<< Signal the thread loop to stop
        mThread.join(); // <<<< Wait for that thread to end
    }

protected:
    void push_impl(const CommandPtr &cmd) { 
        _cmd_queue.push(cmd);
        _cv.notify_one();
    }

private:
    void execute_queue()
    {        
        while (mRun == true)
        {            
            {
                std::unique_lock<std::mutex> lk(_mutex); // mutex gets freed when wait is waiting, otherwise it is blocked.
                _cv.wait(lk, [=] { return !_cmd_queue.empty(); });            
            }
            while (!_cmd_queue.empty())
            {
                CommandPtr cmd;
                if (!_cmd_queue.pop(cmd))
                    break;
                auto cpu_cmd = std::dynamic_pointer_cast<CPUCommand>(cmd);
                cpu_cmd->execute();
            }
            back_to_idle();
        }
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

