//
// Created by rancohen on 23/5/2021.
//
// this file contains the tensor object definition, including tensor views
#pragma once

#ifndef LIBTOSA_TOSA_TENSOR_IMPL_H
#define LIBTOSA_TOSA_TENSOR_IMPL_H
#include <list>
#include <memory>

#include "tosa_types.h"
#include "tosa_memory.h"
#include "tosa_errors.h"
#include "tosa_utils.h"
#include "tosa_stream.h"

namespace libtosa {    
    /**
     * TensorImpl is an Immutable object that describes a tensor or a sub view of another tensor
     * Shapes (and Strides) are Framework order, meaning the last dim is changing fastest in memory
     */
    class TensorImpl;
    typedef std::shared_ptr<TensorImpl> TensorPtr;

    class TensorImpl: public std::enable_shared_from_this<TensorImpl> {
        friend class Tensor;
    public:
        typedef std::list<std::weak_ptr<TensorImpl>> WaekTensorList;
        typedef WaekTensorList::iterator WaekTensorItem;

        explicit TensorImpl(const Shape &shape, DType dtype, const WorkspacePtr &workspace);
        explicit TensorImpl(const Shape &shape, const Shape &stride, DType dtype, const WorkspacePtr &workspace);
        explicit TensorImpl(const TensorPtr &base, const TensorRange &t_range);
        ~TensorImpl();
        
        // Shapes (and Strides) are Framework order, meaning the last dim is changing fastest in memory
        inline const Shape &shape() const { return _shape; }
        inline unsigned rank() const { return _shape.size(); }
        // Element stride, not bytes, last value is always 1
        inline const Shape &stride() const { return _stride; }
        inline const Shape &start_pos() const { return _start_pos; }

        
        TensorPtr subrange(const TensorRange &tr){        
            TensorPtr me = shared_from_this();
            auto ret = std::make_shared<TensorImpl>(me, tr);
            register_as_view(ret);
            return ret;
        }

        bool is_view_overlap(const TensorPtr &sibling_view);

        size_t get_pos_offset(const Shape &pos); // in elements units
        void set_signal(const std::shared_ptr<Signal> &signal, bool from_view = false, bool from_peer = false);
        CommandPtr getWaitIfNotReady();
        void mark_not_ready();
        void sync();
        
        template<typename... Args>
        inline void *get(Args... p) {
            int i=sizeof...(p)-1;            
            size_t offset=__adder( _stride[i--]*p...); // pack goes in reverse...
            sync();
            return (char*)(_memory->ptr())+ _element_size*(_base_offset+offset);
        }

    private:
        size_t _element_size;
        DType _dtype;
        Shape _shape;
        Shape _stride;
        Shape _start_pos;
        std::mutex _signal_mutex;
        
        TensorPtr _view_base;
        WaekTensorList _overlap_tensors;
        WaekTensorList _views;        
        std::list<WeakListReference<TensorImpl>> _my_refernces;
        std::shared_ptr<Signal> _signal;

        size_t _base_offset=0;
        MemoryBlockPtr  _memory;

        void remove_overlap(TensorImpl *peer);
        void register_as_view(const TensorPtr &view);
        void add_me_to(WaekTensorList &list)
        {
            _my_refernces.emplace_back(&list, shared_from_this());
        }
    };
}
#endif //LIBTOSA_TOSA_TENSOR_H
