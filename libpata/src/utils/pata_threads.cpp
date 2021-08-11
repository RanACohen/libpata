#include "pata_threads.h"
#include "pata_debug.h"

ThreadPool global_thread_pool;

ThreadPool::ThreadPool():tid(1)
{
    int num_threads = std::thread::hardware_concurrency();

    for (int i = 0; i < num_threads; i++)
    {
        _the_pool.push_back(std::thread([=] {process_jobs();}));
    }
}


void ThreadPool::process_jobs()
{
    set_local_thread_id(tid++);

    while (!_stop_pool)
    {
        Job job;
        {
            std::unique_lock<std::mutex> lock(_job_q_mx);

            _job_q_cv.wait(lock, [this]()
                           { return !_job_q.empty() || _stop_pool; });
            if (_stop_pool)
                break;
            if (!_job_q.empty())
            {
                job = _job_q.front();
                _job_q.pop_front();
            }
        }
        job();
    }
}

void ThreadPool::Shutdown() // irreversbale, throw this object and create a new pool if you regret
{
    {
        std::unique_lock<std::mutex> lock(_job_q_mx);
        _stop_pool = true;

        _job_q_cv.notify_all(); // wake up all threads.
    }
    // Join all threads.
    for (std::thread &thr : _the_pool)
    {
        thr.join();
    }
    _the_pool.clear();
}
