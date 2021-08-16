#include <memory>
#include <thread>
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
StopWatch system_watch;

thread_local unsigned int thread_local_id = 0; 

void set_local_thread_id(unsigned id)
{
    thread_local_id = id;
}

std::ostream &LOG()
{
    
    return std::cout << "[" << thread_local_id <<" @" << system_watch << "] ";
}

std::ostream& operator<<(std::ostream& os, const FlushLog& flush)
{
    os.flush();
    return os;
}

template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    int size_s = std::snprintf( nullptr, 0, format.c_str(), args ... ) + 1; // Extra space for '\0'
    if( size_s <= 0 ){ throw std::runtime_error( "Error during formatting." ); }
    auto size = static_cast<size_t>( size_s );
    auto buf = std::make_unique<char[]>( size );
    std::snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); // We don't want the '\0' inside
}

std::ostream& operator<<(std::ostream& os, const StopWatch& watch)
{
    auto nsec = watch.leap_nsec();

    auto usec = nsec/1000;
    if (usec == 0)
        return os << nsec << " nSec";
    nsec -= usec*1000;
    auto mSec = usec/1000;
    if (mSec==0)
        return os << string_format("%d.%03d uSec", usec, nsec);
    usec -= mSec*1000;
    auto sec = mSec /1000;
    if (sec==0)
        return os << string_format("%d.%03d mSec", mSec, usec);
    mSec -= sec*1000;
    auto min = sec/60;
    usec += mSec*1000;
    if (min==0)
        return os << string_format("%d.%06d sec", sec, usec);
    sec -= min*60;
    auto hour = min/60;
    if (hour==0)
        return os << string_format("%d:%02d.%06d min", min, sec, usec);
    min -= hour*60;
    auto days = hour/24;
    if (days==0)
        return os << string_format("%d:%02d:%02d.%06d hours", hour, min, sec, usec);
    hour -= days*24;
    return os << string_format("%dd + %d:%02d:%02d.%06d", days, hour, min, sec, usec);
}

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
        auto period =  system_watch.leap_nsec();
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
