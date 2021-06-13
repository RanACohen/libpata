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

using namespace std;

namespace libtosa {    
    class Attr {
        public:
            explicit Attr(DType dtype, std::string name) : _dtype(dtype), _name(name) {}
        private: 
            DType _dtype;
            std::string _name;
    };
    class Tensor;

    typedef std::vector<Tensor> TensorsList;
    typedef std::vector<Attr> AttrList; 

    class Operator{
        public:
            Operator(const std::string &name) : 
                _name(name) {}
            Operator(const std::string &name, 
                    const TensorsList &in,
                    const TensorsList &out, 
                    const AttrList &attr):
                _name(name),
                _inputs (in), 
                _outputs(out),
                _attributes(attr) {}

            Operator(const std::shared_ptr<Operator> &op);
            Operator(const Operator &base);
            
            inline std::string name() { return _name; }
            TensorsList outputs() { return _outputs;}
        protected:
            std::string _name;
            TensorsList _inputs;
            TensorsList _outputs;
            AttrList _attributes; 
    };    
    typedef std::weak_ptr<Operator> OperatorPtr;

    class Signal : public Operator {
        private:
            std::vector<OperatorPtr> _wait_ops;
        public:
            Signal(const std::string& name): Operator(name){}
            void push_back(OperatorPtr& op) { _wait_ops.push_back(op); }
    };

    void schedule(const std::string &op_name, const TensorsList &inputs, const TensorsList &outputs, const AttrList &attributes);
    
    class KernelFunction
    {
    public:
        KernelFunction(const std::string &code);
        KernelFunction(const char *code);
    };

    
    void parallel_for(const Range &index, const KernelFunction &func);
    
    //===----------------------------------------------------------------------===//
    //
    // The functions below define the operation set for the TOSA dialect as defined in
    // the TOSA specfication (https://developer.mlplatform.org/w/tosa/).
    // Each will create tensors and submit for scheduling into execution streams.
    //
    //===----------------------------------------------------------------------===//
    Tensor reluN(const Tensor& in);
    Tensor abs(const Tensor& in);
};

#endif //LIBTOSA_TOSA_OPERATOR_H