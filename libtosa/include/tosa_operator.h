//
// Created by galstar on 31/5/2021.
//
#pragma once
#ifndef LIBTOSA_TOSA_OPERATOR_H
#define LIBTOSA_TOSA_OPERATOR_H
#include <memory>
#include <queue>
#include <vector>
#include <string>
#include <iostream>

#include "tosa_tensor.h"
#include "ThreadPool.h"

const int POOL_SIZE = 4;

using namespace std;

namespace libtosa {    
    class Attr {
        public:
            explicit Attr(DType dtype, std::string name) : _dtype(dtype), _name(name) {}
        private: 
            DType _dtype;
            std::string _name;
    };    
    typedef std::vector<Tensor> TensorsList;
    typedef std::vector<Attr> AttrList; 

    void schedule(std::string op_name, TensorsList inputs, TensorsList outputs, AttrList attributes) {
        static ThreadPool pool (POOL_SIZE);
        std::vector< std::future<int> > results;
        results.emplace_back(
            pool.enqueue([op_name] {
                std::cout << "running " << op_name << std::endl;
                // todo: should be actually performing the operation, by taking the inputs + attributes,
                //       compute and write output - and update it's "ready" bool member. 
                return 0;
            })
        );
    }
    
    //===----------------------------------------------------------------------===//
    //
    // The functions below define the operation set for the TOSA dialect as defined in
    // the TOSA specfication (https://developer.mlplatform.org/w/tosa/).
    // Each will create tensors and submit for scheduling into execution streams.
    //
    //===----------------------------------------------------------------------===//
    Tensor reluN(const Tensor& in){
        Tensor out(in.shape(), in.dtype(), in.workspace());

        AttrList attributes; 
        attributes.push_back(Attr(INT64, "max_int"));
        attributes.push_back(Attr(FLOAT, "max_fp"));

        schedule("tosa.reluN", {in}, {out}, attributes);
        return out;
    }

    Tensor abs(const Tensor& in){
        AttrList attributes; 
        Tensor out(in.shape(), in.dtype(), in.workspace());

        schedule("tosa.abs", {in}, {out}, attributes);
        return out;
    }
};

#endif //LIBTOSA_TOSA_OPERATOR_H