#include <iostream>
#include <mutex>
#include <vector>
#include <thread>
#include <unistd.h>

#include "cpu_backend.h"
#include "tosa_errors.h"

using namespace libtosa;
using namespace libtosa::impl;

class CPUCommand: virtual public Command
{
    public:
        virtual ~CPUCommand() = default;
        virtual void execute() = 0;
};


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
        mThread = std::thread([=] {execute_queue();});
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
    void push_impl(const CommandPtr &cmd) { 
        auto cpu_cmd = std::dynamic_pointer_cast<CPUCommand>(cmd);
        if (!cpu_cmd)
        {
            TOSA_ASSERT(cpu_cmd && "pushing not a CPU command!");
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
                _cv.wait(lk, [=] { return !mRun || !_cmd_queue.empty(); });      
                if (!mRun) break;
            }
            //std::cout << "Stream " << id() << " proceesing queue \n";
            while (!_cmd_queue.empty())
            {
                CommandPtr cmd;
                if (!_cmd_queue.pop(cmd))
                    break;
                auto cpu_cmd = std::dynamic_pointer_cast<CPUCommand>(cmd);
                TOSA_ASSERT(cpu_cmd && "not a CPU command!");
                cpu_cmd->execute();
            }
            //std::cout << "Stream " << id() << " queue Idle... \n";
            back_to_idle();
        }
        std::cout << "Stream " << id() << " exiting. \n";
    }
};


Stream *CPUBackend::createStream(int id) {
    return new CPUStream(id);
}

class CPUComputeCmd: virtual public ComputeCmd, CPUCommand
{
    public:
        CPUComputeCmd(const std::string &name): ComputeCmd(name) {}
        CPUComputeCmd(const std::string &name, 
                const TensorsList &in,
                const TensorsList &out, 
                const AttrList &attr):
            ComputeCmd(name, in, out, attr) {}

        virtual void execute() {
            std::cout << " excuting " << _name << std::endl;
        }
};

class CPUSignal: virtual public Signal, CPUCommand
{
    std::condition_variable _cv;
    std::mutex _mutex;
    public:
        void wait();
        virtual void execute();
        virtual std::shared_ptr<Wait> getWaitCmd();
};


class TestCommand: virtual public Command, CPUCommand
{
    int *_var;
    int _test_val;
    int _msec_sleep;
    public:
        TestCommand(int *variable, int test_val, int sleep_ms=0):
        _var(variable), _test_val(test_val), _msec_sleep(sleep_ms){}

        virtual void execute() {
            usleep(_msec_sleep*1000);             
            *_var = _test_val;
        }
};


CommandPtr CPUBackend::createComputeCmd(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs, const AttrList &attributes)
{
    return std::make_shared<CPUComputeCmd>(op_name, inputs, outputs, attributes);
}

std::shared_ptr<Signal> CPUBackend::createSignal()
{
    return std::make_shared<CPUSignal>();
}

CommandPtr CPUBackend::createTestCmd(int *variable, int test_val, int sleep_ms)
{
    return std::make_shared<TestCommand>(variable, test_val, sleep_ms);
}

    class CPUWait: virtual public Wait, CPUCommand
    {
        public:
        CPUWait(const std::shared_ptr<Signal>& wait_on): Wait(wait_on){}
        virtual void execute();
    };

//todo: seprate to .cpp and .h file for cpu commands.

void CPUWait::execute()
{
    Signal *ps = _wait_on.get();
    auto sig = dynamic_cast<CPUSignal *>(ps);
    sig->wait();
}

void CPUSignal::wait()
{
    std::unique_lock<std::mutex> lk(_mutex);
    _cv.wait(lk, [=]
             { return is_ready(); });
}
void CPUSignal::execute()
{
    signal();
    _cv.notify_all();
}
std::shared_ptr<Wait> CPUSignal::getWaitCmd()
{
    return std::make_shared<CPUWait>(std::dynamic_pointer_cast<Signal>(shared_from_this()));
}
