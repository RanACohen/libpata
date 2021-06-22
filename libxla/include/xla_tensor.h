//
// Created by rancohen on 23/5/2021.
//
// this file contains the tensor object definition, including tensor views
#pragma once

#ifndef LIBXLA_XLA_TENSOR_H
#define LIBXLA_XLA_TENSOR_H
#include <list>
#include <memory>

#include "xla_tensor_impl.h"

namespace libxla {         
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

        void *base_addr() const { return _impl->base_addr(); }
        inline size_t volume() const { return _impl->volume(); }

        inline void mark_not_ready() { _impl->mark_not_ready();}
        inline bool is_ready() { return !_impl->_signal || _impl->_signal->is_ready(); }
        inline void sync() const { return _impl->sync(); }
        inline bool is_contiguous() const { return _impl->is_contiguous(); }

        inline CommandPtr getWaitIfNotReady() { return _impl->getWaitIfNotReady(); }
        inline CommandPtr get_signal_cmd() { return _impl->_signal; }
        

        Tensor operator+(const Tensor &rhs) const;
    };
}
#endif //LIBXLA_XLA_TENSOR_H
