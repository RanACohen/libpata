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

using namespace std;

namespace libpata {   
    typedef std::vector<std::chrono::microseconds> ScheduleTimeMeasurement;
    extern ScheduleTimeMeasurement schedule_time_map;
     
    class Tensor;

    typedef std::vector<Tensor> TensorsList;
 
    class ComputeCmd: public Command {
        public:
            //ComputeCmd(const std::string &name): _name(name) {}
            ComputeCmd(const std::string &name, 
                    const TensorsList &in,
                    const TensorsList &out): 
                Command(name),
                _inputs (in), 
                _outputs(out)
                {
                }

            void mark_output_not_ready()         
            {
                for (auto &o: _outputs) {
                    o.mark_not_ready();
                    add_signal(o.get_signal_cmd());
                } 
            }
            
            inline const std::string &name() { return _cmd_name; }
            TensorsList &outputs() { return _outputs;}
            const TensorsList &inputs() const { return _inputs;}

        protected:           
            //std::string _name;
            TensorsList _inputs;
            TensorsList _outputs;            
    };    
    typedef std::shared_ptr<ComputeCmd> ComputeCmdPtr;
    
    void schedule(const ComputeCmdPtr &cmd);


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

    /**
     * Add2D: does parallel Add operation of out=A+B, 
     * inA - Input A Matrix, must be a 2D Tensor
     * inB - Input B Matrix, must be a 2D Tensor
     * out - output Matrix (already allocated) of size (A-rows,B-cols)
     * outViews - an empty Tensor list of views to be placed after the split according to a h/w friendly split
     * block_size - the block size to partition the rows into parallel blocks/views
     * */
    void Add2D(Tensor& inA, Tensor& inB, Tensor& out, TensorsList &outViews, int block_size=10); 

    /**
     * Add2D: does parallel Add operation of out=A+B, 
     * inA - Input A Tesnor
     * inB - Input B Tesnor
     * out - output Tesnor (already allocated) of smae size as inputs     
     * */
    void Add(const Tensor& inA, const Tensor& inB, Tensor& out); 

    bool test_Libxsmm(const Tensor& inA, const Tensor& inB, Tensor& out);
};

#endif //LIBPATA_PATA_OPERATOR_H