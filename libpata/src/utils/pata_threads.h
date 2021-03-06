#pragma once
#ifndef LIBPATA_THREADS_H
#define LIBPATA_THREADS_H
#include <thread>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <atomic>
#include <stack>

typedef std::function<void()> Job;

class ThreadPool
{
    std::vector<std::thread> _the_pool;
    bool _stop_pool = false;
    std::mutex _job_q_mx;
    std::condition_variable _job_q_cv;
    std::deque<Job> _job_q;
    std::atomic<unsigned int> tid;

public:
    ThreadPool();
    ~ThreadPool()
    {
        Shutdown();
    }

    void ExecuteJob(const Job &a_Job)
    {
        std::unique_lock<std::mutex> lock(_job_q_mx);
        _job_q.push_back(a_Job);
        _job_q_cv.notify_one();
    }

    void Shutdown(); // irreversbale, throw this object and create a new pool if you regret

private:
    void process_jobs();
};

extern ThreadPool global_thread_pool;

template<typename T>
class ObjectPool
{
    std::stack<T*> _pool;
    std::mutex _mx;
    unsigned int _given_obj_cnt;

public:
    ObjectPool(int size=256):_given_obj_cnt(0)
    {
        for(int i=0; i<size; i++) _pool.push(new T());
    }

    std::shared_ptr<T> get()
    {
        std::unique_lock<std::mutex> lock(_mx);
        _given_obj_cnt++;
        T* obj;
        if (_pool.empty()) {
            obj = new T();
        }
        else {
            obj = _pool.top();
            _pool.pop();
        } 
        
        return std::shared_ptr<T>(obj, [=](auto p) {return_obj(p);});
    }

private:
    void return_obj(T* obj)
    {
        obj->clear();
        _given_obj_cnt--;
        std::unique_lock<std::mutex> lock(_mx);
        _pool.push(obj);
    }
};

#endif //LIBPATA_THREADS_H