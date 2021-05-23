//
// Created by rancohen on 23/5/2021.
//
// this file contains the tensor object definition, including tensor views

#ifndef LIBTOSA_TOSA_TENSOR_H
#define LIBTOSA_TOSA_TENSOR_H
#include <vector>
#include <memory>

#include "tosa_types.h"

namespace libtosa {

    /**
     * Tensor is an Immutable object that describes a tensor or a sub view of another tensor
     * Shapes (and Strides) are Framework order, meaning the last dim is changing fastest in memory
     */
    class Tensor {
    public:
        explicit Tensor(const Shape &shape, DType dtype){
            _dtype = dtype;
            _shape = shape;
            _stride.push_back(dtype_byte_size(dtype));
        }
        explicit Tensor(const Shape &shape, const Shape &stride, DType dtype){
            _dtype = dtype;
            _shape = shape;
            _stride = stride;
        }

        const Shape &shape() const { return _shape; }
        const Shape &stride() const { return _stride; }

    private:
        DType _dtype;
        Shape _shape;
        Shape _stride;
        std::shared_ptr<Tensor> _view_base;
    };

}
#endif //LIBTOSA_TOSA_TENSOR_H
