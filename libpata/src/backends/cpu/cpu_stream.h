//
// Created by rcohen on 20/6/2021.
//
#pragma once
#ifndef LIBPATA_PATA_CPU_STREAM_H
#define LIBPATA_PATA_CPU_STREAM_H
#include <thread>

#include "pata_stream.h"

namespace libpata
{
    namespace impl
    {

        template <class T>
        struct MutexFreeQueue
        {
            int _size;
            int _get;
            int _put;
            T *_cyclic_buffer;

            MutexFreeQueue(int size)
            {
                _size = size;
                _cyclic_buffer = new T[size];
                _get = 0;
                _put = 0;
            }
            ~MutexFreeQueue()
            {
                delete[] _cyclic_buffer;
            }
            void push(const T &item)
            {
                auto next_put = (_put + 1) % _size;
                PATA_ASSERT(next_put != _get); // queue if full, nothing to do, bail out....
                _cyclic_buffer[_put] = item;
                _put = next_put;
            }
            bool pop(T &ret)
            {
                if (empty())
                    return false;
                ret = _cyclic_buffer[_get];
                _get = (_get + 1) % _size;
                return true;
            }

            inline bool empty() const { return _get == _put; }
        };

        class CPUStream : public Stream
        {
            friend class StreamPool;
            MutexFreeQueue<CommandPtr> _cmd_queue;
            bool mRun; // Use a race condition safe data
                       // criterium to end that thread loop
            std::thread mThread;
            std::condition_variable _cv;
            std::mutex _mutex;

        public:
            CPUStream(int id) : Stream(id), mRun(true), _cmd_queue(1024)
            {
                mThread = std::thread([=]
                                      { execute_queue(); });
            }
            ~CPUStream()
            {
                std::cout << "Stream " << id() << " closing...\n";
                mRun = false; // <<<< Signal the thread loop to stop
                _cv.notify_one();
                mThread.join(); // <<<< Wait for that thread to end
                std::cout << "Stream " << id() << " ended\n";
            }

        protected:
            void push_impl(const CommandPtr &cmd)
            {
                auto cpu_cmd = std::dynamic_pointer_cast<CPUCommand>(cmd);
                if (!cpu_cmd)
                {
                    PATA_ASSERT(cpu_cmd && "pushing not a CPU command!");
                }
                _cmd_queue.push(cmd);
                _cv.notify_one();
            }

        private:
            void execute_queue()
            {
                std::cout << "Stream " << id() << " running.. \n";
                while (mRun == true)
                {
                    {
                        std::unique_lock<std::mutex> lk(_mutex); // mutex gets freed when wait is waiting, otherwise it is blocked.
                        _cv.wait(lk, [=]
                                 { return !mRun || !_cmd_queue.empty(); });
                        if (!mRun)
                            break;
                    }
                    //std::cout << "Stream " << id() << " proceesing queue \n";
                    while (!_cmd_queue.empty())
                    {
                        CommandPtr cmd;
                        if (!_cmd_queue.pop(cmd))
                            break;
                        auto cpu_cmd = std::dynamic_pointer_cast<CPUCommand>(cmd);
                        PATA_ASSERT(cpu_cmd && "not a CPU command!");
                        //std::cout << "Executing cmd... on stream " << id() << "\n";
                        cpu_cmd->execute();
                    }
                    //std::cout << "Stream " << id() << " queue Idle... \n";
                    back_to_idle();
                }
                std::cout << "Stream " << id() << " exiting. \n";
            }
        };

    }
}

#endif // LIBPATA_PATA_CPU_STREAM_H