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

#include "tosa_tensor.h"

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
        // todo: Gal implement me. 
        //
        // This schedules the work into streams, assign it to new/existing stream 
        // and signal/wait. 
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