#include <unistd.h>
#include "cpu_commands.h"

using namespace libtosa;
using namespace libtosa::impl;

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

void TestCommand::execute()
{
    usleep(_msec_sleep * 1000);
    *_var = _test_val;
}


 void CPUAddCmd::execute()
 {
     //todo: implement me
     std::cout << "add two tensors here please...\n";
 }