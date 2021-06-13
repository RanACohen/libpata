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
#include "tosa_operator.h"
#include "tosa_errors.h"

namespace libtosa {
    static inline size_t __adder(size_t v) { return v; }
    template<typename T, typename... Args>
    static inline size_t __adder(T base, Args... args) {        
        return base + __adder(args...);
    }

    class Tensor;
    class Signal;
    /**
     * TensorImpl is an Immutable object that describes a tensor or a sub view of another tensor
     * Shapes (and Strides) are Framework order, meaning the last dim is changing fastest in memory
     */
    class TensorImpl {
        friend class Tensor;
    public:
        explicit TensorImpl(const Shape &shape, DType dtype, const WorkspacePtr &workspace);
        explicit TensorImpl(const Shape &shape, const Shape &stride, DType dtype, const WorkspacePtr &workspace);
        explicit TensorImpl(const std::shared_ptr<TensorImpl> &base, const TensorRange &t_range);

        // Shapes (and Strides) are Framework order, meaning the last dim is changing fastest in memory
        inline const Shape &shape() const { return _shape; }
        inline unsigned rank() const { return _shape.size(); }
        // Element stride, not bytes, last value is always 1
        inline const Shape &stride() const { return _stride; }

        size_t get_pos_offset(const Shape &pos); // in elements units
        inline void set_signal(std::shared_ptr<Signal> &signal, bool from_view = false) { 
            TOSA_ASSERT(!_signal); // if this tensor is not ready we cannot overide it
            _signal = signal;
            if (_view_base) // if I am a view so it my base not ready
                _view_base->set_signal(signal);

            // set the signal event on all the other tensors that are my view
            //if the base is not ready, so are all the views
            // only if not coming from a view
            if (from_view) return;
            for (const auto &v : _views)
            {
                auto v_shrd = v.lock();
                if (!v_shrd) continue;
                v_shrd->set_signal(signal); 
            }
        }

        template<typename... Args>
        inline void *get(Args... p) {
            int i=sizeof...(p)-1;            
            size_t offset=__adder( _stride[i--]*p...); // pack goes in reverse...
            return (char*)(_memory->ptr())+ _element_size*(_base_offset+offset);
        }

    private:
        size_t _element_size;
        DType _dtype;
        Shape _shape;
        Shape _stride;
        
        std::shared_ptr<TensorImpl> _view_base;
        std::vector<std::weak_ptr<TensorImpl>> _views;
        std::shared_ptr<Signal> _signal;

        size_t _base_offset=0;
        MemoryBlockPtr  _memory;
    };

    // Wrap it publicly so users can treat is regular object and pass it via value and create temp in stack
    class Tensor
    {
        std::shared_ptr<TensorImpl> _impl;
    public:
        explicit Tensor(const Shape &shape, DType dtype, const WorkspacePtr &workspace):
            _impl(std::make_shared<TensorImpl>(shape, dtype, workspace)) {}
        explicit Tensor(const Shape &shape, const Shape &stride, DType dtype, const WorkspacePtr &workspace):
                _impl(std::make_shared<TensorImpl>(shape, stride, dtype, workspace)) {}
        explicit Tensor(const Tensor &base, const TensorRange &t_range):
                _impl(std::make_shared<TensorImpl>(base._impl, t_range)) {
                    base._impl->_views.push_back(_impl);
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
        T* at(size_t... p0) const {return (T*)_impl->get(p0...);}

        inline void set_signal(std::shared_ptr<Signal> &signal) { _impl->set_signal(signal);}
        inline std::shared_ptr<Signal>& signal() { return _impl->_signal; }
        inline const bool is_ready() { return !_impl->_signal;} // todo: think maybe it should lock and create a new wait

        Tensor operator+(const Tensor &rhs) const;
    };
}
#endif //LIBTOSA_TOSA_TENSOR_H
