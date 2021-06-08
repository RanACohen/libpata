//
// Created by rcohen on 08/06/2021.
//
#pragma once
#ifndef LIBTOSA_TOSA_STREAM_HPP
#define LIBTOSA_TOSA_STREAM_HPP
#include <memory>

namespace libtosa {  

    class StreamPool;

    class Stream {
        friend class StreamPool;
        int _id;
        Stream(int id);

    public:
        int id() { return _id;}
    };

    typedef std::shared_ptr<Stream> StreamPtr;

    class StreamManager {
        public:
            static StreamManager &Inst();

            StreamPtr createStream(); 

    private:
        std::shared_ptr<StreamPool> _pool;
        StreamManager();
    };

};

#endif //LIBTOSA_TOSA_STREAM_HPP
