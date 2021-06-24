#include "pata_debug.h"

struct dead_lock_debug_item {
    std::chrono::high_resolution_clock::time_point log_time;
    int wait_str_id;
    int cmd_id;
    int signal_in_str_id;
    EventType event;
};
#define DEADLOCK_LOG_SIZE 65536
dead_lock_debug_item dead_lock_debug_info[DEADLOCK_LOG_SIZE];
std::atomic<size_t> deadlock_put_index(0);
std::chrono::high_resolution_clock::time_point boot_time = std::chrono::high_resolution_clock::now();

void deadlock_debug_reset()
{
    deadlock_put_index=0;
}

void trace_stream_wait(int sid, size_t from)
{   
    if (sid<0) return;
    for (int j=from; j>=0; j--)
    {
        auto &item = dead_lock_debug_info[j];
        if (item.wait_str_id == sid)
        {            
            if (item.event == EventType::WAIT_WOKE_UP)
            {
                std::cout << "\t[ at " << j << "] str " << sid << " ... eventually woke up"  << std::endl;
                return;
            }
            if (item.event != EventType::STREAM_WAIT) continue;
            std::cout << "\t[ at " << j << "] str " << sid << " ... is waiting on stream " << item.signal_in_str_id << std::endl;
            trace_stream_wait(item.signal_in_str_id, j+1);
            return;
        }
    }        
};


void dump_dead_lock()
{
    size_t last = deadlock_put_index;
    for (unsigned i=0; i<last; i++)
    {        
        auto &item = dead_lock_debug_info[i];
        auto period =  std::chrono::duration_cast<std::chrono::microseconds>(item.log_time-boot_time).count();
        if (item.event == EventType::WAIT_WOKE_UP)
        {
            std::cout << "[" << i << "] " << period << ": stream #" << item.wait_str_id << " woke up from signal " << item.cmd_id << std::endl;
        } else if (item.event == EventType::STREAM_WAIT) {
            std::cout << "[" << i << "] " << period << ": stream #" << item.wait_str_id << " wait on " << item.cmd_id << " in stream #" << item.signal_in_str_id << std::endl;
            //trace_stream_wait(item.signal_in_str_id, i+1);
        } else if (item.event == EventType::CMD_PUSH) {
            std::cout << "[" << i << "] " << period << ": stream #" << item.wait_str_id << " cmd was pushed "  << item.cmd_id << std::endl;
        } else if (item.event == EventType::BACK_TO_IDLE) {
            std::cout << "[" << i << "] " << period << ": stream #" << item.wait_str_id << " Back to idle "  << std::endl;
        } else if (item.event == EventType::SIGNAL_ON) {
            std::cout << "[" << i << "] " << period << ": stream #" << item.wait_str_id << " signaled signal " << item.cmd_id << std::endl;
        } else if (item.event == EventType::CMD_POP) {
            std::cout << "[" << i << "] " << period << ": stream #" << item.wait_str_id << " popped cmd " << item.cmd_id << std::endl;
        } else {
            std::cout << "[" << i << "] bad event ID!!!!!!!!!!!"<< std::endl;
        }
    }
}

#ifdef ENABLE_EVENT_TRACE
void log_dead_lock(int wait_id, int cmd_id, int sig_str_id, EventType event)
{
    size_t i = (deadlock_put_index++) % DEADLOCK_LOG_SIZE;
    auto &item = dead_lock_debug_info[i];
    item.log_time = std::chrono::high_resolution_clock::now();
    item.wait_str_id = wait_id;
    item.cmd_id = cmd_id;
    item.signal_in_str_id = sig_str_id;        
    item.event = event;
}
#endif
