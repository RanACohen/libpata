//
// Created by rancohen on 23/5/2021.
//
// this file contains the tensor object definition, including tensor views
#pragma once

#ifndef LIBTOSA_TOSA_TENSOR_H
#define LIBTOSA_TOSA_TENSOR_H
#include <vector>
#include <memory>

#include "tosa_types.h"
#include "tosa_memory.h"

namespace libtosa {

    /**
     * Tensor is an Immutable object that describes a tensor or a sub view of another tensor
     * Shapes (and Strides) are Framework order, meaning the last dim is changing fastest in memory
     */
    class Tensor;
    typedef std::shared_ptr<Tensor> TensorPtr;

    class Tensor {
    public:
        explicit Tensor(const Shape &shape, DType dtype, const WorkspacePtr &workspace);
        explicit Tensor(const Shape &shape, const Shape &stride, DType dtype, const WorkspacePtr &workspace);
        explicit Tensor(const TensorPtr &base, const TensorRange &t_range);

        // Shapes (and Strides) are Framework order, meaning the last dim is changing fastest in memory
        const Shape &shape() const { return _shape; }
        unsigned rank() const { return _shape.size(); }
        size_t get_pos_offset(const Shape &pos); // in elements units

        // Element stride, not bytes, last value is always 1
        const Shape &stride() const { return _stride; }

    private:
        DType _dtype;
        Shape _shape;
        Shape _stride;
        TensorPtr _view_base;
        RelativeSize _base_offset=0;
        MemoryBlockPtr  _memory;
    };

}
#endif //LIBTOSA_TOSA_TENSOR_H
