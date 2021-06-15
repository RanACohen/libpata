//
// Created by rancohen on 23/5/2021.
//
// this file contains the tensor object definition, including tensor views
#pragma once

#ifndef LIBTOSA_TOSA_TENSOR_H
#define LIBTOSA_TOSA_TENSOR_H
#include <list>
#include <memory>

#include "tosa_types.h"
#include "tosa_memory.h"
#include "tosa_tensor impl.h"
#include "tosa_operator.h"
#include "tosa_errors.h"
#include "tosa_utils.h"

namespace libtosa {
    class Signal;
     
    // Wrap it publicly so users can treat is regular object and pass it via value and create temp in stack
    class Tensor
    {
        typedef std::shared_ptr<TensorImpl> ImplPtr;
        ImplPtr _impl;
        explicit Tensor(const ImplPtr &impl):_impl(impl){}

    public:
        explicit Tensor(const Shape &shape, DType dtype, const WorkspacePtr &workspace):
            _impl(std::make_shared<TensorImpl>(shape, dtype, workspace)) {}
        explicit Tensor(const Shape &shape, const Shape &stride, DType dtype, const WorkspacePtr &workspace):
                _impl(std::make_shared<TensorImpl>(shape, stride, dtype, workspace)) {}
       
        Tensor operator[](const TensorRange &tr){
            return Tensor(_impl->subrange(tr));
        }

        template <typename... R> 
        Tensor subrange(const R&... r){
            return Tensor(_impl->subrange(TensorRange{r...}));
        }

        static inline Tensor like(const Tensor&t) { 
            return Tensor(t.shape(), t.dtype(), t.workspace());
        }

        // access impl private members to avoid nested function calls
        inline const Shape &shape() const { return _impl->_shape; }
        inline DType dtype() const { return _impl->_dtype; }
        inline unsigned rank() const { return _impl->_shape.size(); }
        // Element stride, not bytes, last value is always 1
        inline const Shape &stride() const { return _impl->_stride; }
        inline const WorkspacePtr &workspace() const { return _impl->_memory->workspace(); }

        template<typename T, typename... size_t>
        T* at(size_t... p) const {return (T*)_impl->get(p...);}

        inline void set_signal(std::shared_ptr<Signal> &signal) { _impl->set_signal(signal);}
        inline std::shared_ptr<Signal>& signal() { return _impl->_signal; }
        inline const bool is_ready() { return !_impl->_signal;} // todo: think maybe it should lock and create a new wait

        Tensor operator+(const Tensor &rhs) const;
    };
}
#endif //LIBTOSA_TOSA_TENSOR_H
