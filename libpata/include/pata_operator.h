//
// Created by galstar on 31/5/2021.
//
#pragma once
#ifndef LIBPATA_PATA_OPERATOR_H
#define LIBPATA_PATA_OPERATOR_H
#include <memory>
#include <queue>
#include <vector>
#include <string>
#include <iostream>
#include <map>
#include <chrono>

#include "pata_tensor.h"
#include "pata_stream.h"

using namespace std;

namespace libpata {   
    typedef std::vector<std::chrono::microseconds> ScheduleTimeMeasurement;
    extern ScheduleTimeMeasurement schedule_time_map;
 
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

    class ComputeCmd: virtual public Command {
        public:
            //ComputeCmd(const std::string &name): _name(name) {}
            ComputeCmd(const std::string &name, 
                    const TensorsList &in,
                    const TensorsList &out, 
                    const AttrList &attr):
                _name(name),
                _inputs (in), 
                _outputs(out),
                _attributes(attr) {}
                     
            
            inline const std::string &name() { return _name; }
            TensorsList &outputs() { return _outputs;}
            const TensorsList &inputs() const { return _inputs;}

        protected:
            std::string _name;
            TensorsList _inputs;
            TensorsList _outputs;
            AttrList _attributes; 
    };    
    typedef std::shared_ptr<ComputeCmd> ComputeCmdPtr;
    
    void schedule(const ComputeCmdPtr &cmd);
    
    class KernelFunction
    {
    public:
        KernelFunction(const std::string &code);
        KernelFunction(const char *code);
    };
    

    Tensor reluN(const Tensor& in);
    Tensor abs(const Tensor& in);
    
    /**
     * MatMul: does a Matrix multiplication of out=A*B, 
     * inA - Input A Matrix, must be a 2D Tensor (todo: Batched MatMul for 3D+)
     * inB - Input B Matrix, must be a 2D Tensor
     * out - output Matrix (already allocated) of size (A-rows,B-cols)
     * outViews - an empty Tensor list of views to be placed after the split according to a h/w friendly split
     * */
    void MatMul(const Tensor& inA, const Tensor& inB, Tensor& out, TensorsList &outViews);

    bool test_Libxsmm(const Tensor& inA, const Tensor& inB, Tensor& out);
};

#endif //LIBPATA_PATA_OPERATOR_H