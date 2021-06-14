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
    class TensorImpl: public std::enable_shared_from_this<TensorImpl> {
        friend class Tensor;
    public:
        typedef std::list<std::weak_ptr<TensorImpl>> WaekTensorList;
        typedef WaekTensorList::iterator WaekTensorItem;

        explicit TensorImpl(const Shape &shape, DType dtype, const WorkspacePtr &workspace);
        explicit TensorImpl(const Shape &shape, const Shape &stride, DType dtype, const WorkspacePtr &workspace);
        explicit TensorImpl(const std::shared_ptr<TensorImpl> &base, const TensorRange &t_range);
        ~TensorImpl();
        
        // Shapes (and Strides) are Framework order, meaning the last dim is changing fastest in memory
        inline const Shape &shape() const { return _shape; }
        inline unsigned rank() const { return _shape.size(); }
        // Element stride, not bytes, last value is always 1
        inline const Shape &stride() const { return _stride; }
        inline const Shape &start_pos() const { return _start_pos; }

        
        std::shared_ptr<TensorImpl> subrange(const TensorRange &tr){        
            std::shared_ptr<TensorImpl> me = shared_from_this();
            auto ret = std::make_shared<TensorImpl>(me, tr);
            register_as_view(ret);
            return ret;
        }

        void register_as_view(const std::shared_ptr<TensorImpl> &view);
        void add_me_to(WaekTensorList &list)
        {
            _my_refernces.emplace_back(list, list.insert(list.begin(), shared_from_this()));
        }

        bool is_view_overlap(const std::shared_ptr<TensorImpl> &sibling_view);

        size_t get_pos_offset(const Shape &pos); // in elements units
        void set_signal(std::shared_ptr<Signal> &signal, bool from_view = false, bool from_peer = false);

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
        Shape _start_pos;
        struct WeakListReference {
            WaekTensorList &_list;
            WaekTensorItem _item;

            WeakListReference(WaekTensorList &list, const WaekTensorItem &item):_list(list), _item(item){}
            ~WeakListReference() { _list.erase(_item); }
        };
        
        std::shared_ptr<TensorImpl> _view_base;
        WaekTensorList _overlap_tensors;
        WaekTensorList _views;        
        std::list<WeakListReference> _my_refernces;
        std::shared_ptr<Signal> _signal;

        size_t _base_offset=0;
        MemoryBlockPtr  _memory;
    };

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
