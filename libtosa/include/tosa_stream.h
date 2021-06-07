//
// Created by galstar on 31/5/2021.
//
#pragma once
#ifndef LIBTOSA_TOSA_STREAM_H
#define LIBTOSA_TOSA_STREAM_H
#include <memory>
#include <queue>
#include <vector>

#include "tosa_tensor.h"

namespace libtosa {
    class Stream {
    public:
        explicit Stream();

        void push(const Tensor& tensor) const { _stream.push(tensor); }
        const Tensor &pop() const { return _stream.pop(); }
    private:
        std::queue<Tensor> _stream; 
    };

    class StreamsPool {
    public: 
        explicit StreamsPool();

    private: 
        std::vector<Stream> _pool;
    };

};