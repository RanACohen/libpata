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
    class Tensor;
    typedef std::shared_ptr<Tensor> TensorPtr;

    class Tensor {
    public:
        explicit Tensor(const Shape &shape, DType dtype);
        explicit Tensor(const Shape &shape, const Shape &stride, DType dtype);
        explicit Tensor(const TensorPtr &base, const TensorRange &range);

        // Shapes (and Strides) are Framework order, meaning the last dim is changing fastest in memory
        const Shape &shape() const { return _shape; }

        // Element stride, not bytes, last value is always 1
        const Shape &stride() const { return _stride; }

    private:
        DType _dtype;
        Shape _shape;
        Shape _stride;
        TensorPtr _view_base;
    };

}
#endif //LIBTOSA_TOSA_TENSOR_H
