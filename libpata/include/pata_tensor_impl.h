//
// Created by rancohen on 23/5/2021.
//
// this file contains the tensor object definition, including tensor views
#pragma once

#ifndef LIBPATA_PATA_TENSOR_IMPL_H
#define LIBPATA_PATA_TENSOR_IMPL_H
#include <list>
#include <memory>

#include "pata_types.h"
#include "pata_memory.h"
#include "pata_errors.h"
#include "pata_utils.h"
#include "pata_stream.h"

namespace libpata {    
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
        inline size_t volume() const { return _volume; }
        inline bool is_contiguous() const {
            return _stride[0]*_shape[0] == _volume;
        }

        TensorPtr subrange(const TensorRange &tr){        
            TensorPtr me = shared_from_this();
            auto ret = std::make_shared<TensorImpl>(me, tr);
            register_as_view(ret);
            return ret;
        }

        bool is_view_overlap(const TensorPtr &sibling_view);

        size_t get_pos_offset(const Shape &pos) const; // in elements units
        std::shared_ptr<Wait> getWaitIfNotReady();
        void get_wait_list(const std::shared_ptr<Wait> &wait_list, bool from_view=false, bool from_peer=false);
        bool is_ready(bool check_peers=true, bool check_views=true, bool check_base=true);
        void mark_not_ready();

        void sync();

        inline void *base_addr() const {
            return (char*)(_memory->ptr())+ _element_size*(_base_offset);
        }
        
        template<typename... Args>
        inline void *get(Args... p) {
            int i=sizeof...(p)-1;
            size_t offset=__adder( _stride[i--]*p...); // pack goes in reverse...
            sync();
            return (char*)(_memory->ptr())+ _element_size*(_base_offset+offset);
        }

        template<typename T>
        void fill(T start_val, T step)
        {
            PATA_ASSERT(is_contiguous() && "FIll works only on contigious tensors...");
            T *pData = (T*)base_addr();
            for (unsigned i=0; i<_volume; i++) pData[i]=start_val+ i*step;
        }

    private:
        size_t _element_size;
        DType _dtype;
        Shape _shape;
        size_t _volume;
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
#endif //LIBPATA_PATA_TENSOR_H
