//
// Created by rcohen on 08/06/2021.
//
#pragma once
#ifndef LIBTOSA_TOSA_STREAM_HPP
#define LIBTOSA_TOSA_STREAM_HPP
#include <memory>
#include <queue>

#include "tosa_tensor.h"
#include "tosa_operator.h"

namespace libtosa {  
    class StreamPool;
    typedef std::shared_ptr<Operator> OperatorPtr;

    class Stream {
        friend class StreamPool;
        int _id;
        std::queue<OperatorPtr> _op_queue; 

        Stream(int id);

    public:
        int id() { return _id;}
        void push(OperatorPtr& op); // todo: how do I push a "wait" to a new stream?
        void add_single_op (OperatorPtr& op) { _op_queue.push(op);}
        const OperatorPtr& pop();
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
