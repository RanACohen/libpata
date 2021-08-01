#include <iostream>

#include "pata_debug.h"
#include "pata_tensor.h"
#include "pata_errors.h"
#include "pata_commands.h"
#include "pata_operator.h"
#include "pata_backend.h"

using namespace libpata;

int libpata::dtype_byte_size(DType dtype)
{
    //                            0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    static int DTypeByteSize[] = {0, 4, 1, 1, 2, 2, 4, 8, 1, 1, 2, 8, 4, 8, 8, 16, 2};
    return ((dtype < 0) || (dtype > DType::LAST)) ? -1 : DTypeByteSize[dtype];
}


struct TensorSpace
{
    Shape stride;
    Shape size;
    Shape start;
    unsigned rank;

    TensorSpace() {}
    TensorSpace(TensorImpl *t)
    {
        stride = t->stride();
        size = t->shape();
        start = t->start_pos();
        rank = size.size();
    }

    bool is_overlap(const TensorSpace &sib)
    {
        for (unsigned i = 0; i < rank; i++)
        {
            auto my_end = (size[i] + start[i]) * stride[i];
            auto sib_start = sib.start[i] * sib.stride[i];
            if (my_end <= sib_start)
                return false;
            auto my_start = start[i] * stride[i];
            auto sib_end = (sib.start[i] + sib.size[i]) * sib.stride[i];
            if (my_start >= sib_end)
                return false;
        }
        return true;
    }

    bool is_overlap(const TensorSpace &sib, const Shape &dilation)
    {
        for (unsigned i = 0; i < rank; i++)
        {
            auto my_end = (size[i] + start[i]) * stride[i];
            auto sib_start = sib.start[i] * sib.stride[i];
            if (my_end <= sib_start)
                return false;
            auto my_start = start[i] * stride[i];
            auto sib_end = (sib.start[i] + sib.size[i]) * sib.stride[i];
            if (my_start >= sib_end)
                return false;

            if (dilation[i] == 1)
                continue;
            // see if the start modul the dilation is the same, if so then this dim overlap
            // if not then they interleave and do not overlap
            if ((start[i] % dilation[i]) != (sib.start[i] % dilation[i]))
                return false;
        }
        return true;
    }
};

TensorImpl::TensorImpl(const Shape &shape, DType dtype, const WorkspacePtr &workspace)
{
    _dtype = dtype;
    _shape = shape;
    _element_size = dtype_byte_size(dtype);
    size_t s = 1;
    _stride.push_back(1);    
    for (unsigned i = shape.size() - 1; i > 0; --i)
    {
        s *= shape[i];
        _stride.insert(_stride.begin(), s);
    }
    s *= shape[0];
    _volume = s;
    _memory = MemoryBlock::allocate(s* dtype_byte_size(dtype), workspace);
}

/**
 * Allocate with custom strides - this is the developer responsibility, should not be used really.
 * use Allocator with Range for custom strides
 * @param shape
 * @param stride
 * @param dtype
 * @param workspace
 */
TensorImpl::TensorImpl(const Shape &shape, const Shape &stride, DType dtype, const WorkspacePtr &workspace)
{
    _dtype = dtype;
    _shape = shape;
    _stride = stride;
    _element_size = dtype_byte_size(dtype);
    _volume = 1;
    for (auto s:shape)
    {
        _volume*=s;
    }

    auto s = stride[0] * shape[0] * dtype_byte_size(dtype);
    _memory = MemoryBlock::allocate(s, workspace);
}

#define CLIP(v, s, e) v = v < (s) ? (s) : v > (e) ? (e) \
                                                  : v

TensorImpl::TensorImpl(const TensorPtr &base, const TensorRange &t_range)
{
    _dtype = base->_dtype;
    _view_base = base;
    auto base_shape = base->shape();
    _stride = base->stride();
    _memory = base->_memory;
    _element_size = base->_element_size;
     _volume = 1;
    for (unsigned i = 0; i < base->rank(); i++)
    {
        auto range = i < t_range.size() ? t_range[i] : Range();
        // handle negative like in python: if end ==0 means the end as -1 is one before (unlike start).
        auto end = range.end > 0 ? range.end : base_shape[i] + range.end;
        auto start = range.start >= 0 ? range.start : base_shape[i] + range.start;
        // make sure no out of bound
        CLIP(end, 0, base_shape[i]);
        CLIP(start, 0, base_shape[i]);
        auto s = (end - start + range.step - 1) / range.step;
        _volume *= s;
        _shape.push_back(s); // ceil: step 7 in 22 :4=ciel(22/7) 0,7,14,21
        if (range.step != 1)
        {
            _stride[i] *= range.step;
        }
        _start_pos.push_back(start);
    }
    _base_offset = base->get_pos_offset(_start_pos);    
}

TensorImpl::~TensorImpl()
{
    //LOG() << " Destroying tensor\n";
    {
        std::unique_lock<std::mutex> lk(_overlap_guard);

        for (auto &ot : _overlap_tensors)
        {
            TensorPtr t = ot.lock();
            if (t)
                t->remove_overlap(this);
            else
            {
                std::cout << "oops!\n";
            }
        }
    }

    if (_view_base)
    {
        _view_base->remove_view(_my_refernces_to_base);
    }
}

void TensorImpl::register_overlap(const TensorPtr &view)
{
     std::unique_lock<std::mutex> lk(_overlap_guard);
    _overlap_tensors.push_back(view);
}

void TensorImpl::register_view(const TensorPtr &view)
{
     std::unique_lock<std::mutex> lk(_views_guard);
    // check out views overlapping
    // need special handling view->_overlap_tensors.push_back(shared_from_this()); // base is also overlapping... no need to register re, base will not get free with views
    for (auto &i : _views)
    {
        TensorPtr vi = i.lock();
        if (!vi)
        {
            continue;
        } // should not happen as view gets unregister in destructor

        if (vi->is_view_overlap(view))
        {
            vi->register_overlap(view);
            view->register_overlap(vi);
        }
    }    
    view->_my_refernces_to_base._item = _views.insert(_views.end(), view);    
}

void TensorImpl::remove_view(const WeakListReference<TensorImpl> &ref)
{
    std::unique_lock<std::mutex> lk(_views_guard);
    _views.erase(ref._item);
}

void TensorImpl::remove_overlap(TensorImpl *peer)
{
    std::unique_lock<std::mutex> lk(_overlap_guard);

    WaekTensorItem item = _overlap_tensors.begin();
    while (item != _overlap_tensors.end())
    {
        TensorPtr t = item->lock();
        if (!t || t.get()==peer)
        {
            item = _overlap_tensors.erase(item);
        } else {
            item++;
        }
    }
}



void TensorImpl::mark_not_ready()
{
    _signal = BackendManager::Inst().backend()->createSignal();
}


bool TensorImpl::is_ready(bool check_peers, bool check_views, bool check_base)
{
    if (_signal && !_signal->is_ready())
        return false;

    if (check_base && _view_base) // if I am a view and base is not ready , so am I. but no need to check base views, I am checking overlaps
        if (!_view_base->is_ready(true, false, true)) 
            return false;

    if (check_peers)
    {
        // if any overlap is not ready, so am I
        std::unique_lock<std::mutex> lk(_overlap_guard);

        for (const auto &v : _overlap_tensors)
        {
            auto vi = v.lock();
            if (!vi)
                continue;
            if (!vi->is_ready(false, true, false)) // peer views might overlap
                return false; 
        }
    }

    // If any of my views is not ready, so am I
    if (check_views) 
    {
        std::unique_lock<std::mutex> lk(_views_guard);    
        for (const auto &v : _views)
        {
            auto vi = v.lock();
            if (!vi)
                continue;
            if (!vi->is_ready(false, true, false)) // dont check base (infinite loop) and dont check peers
                return false;
        }               
    }
    return true;
}

void TensorImpl::get_wait_list(const std::shared_ptr<Wait> &wait, bool from_view, bool from_peer)
{
    if (_signal && !_signal->is_ready())
    {
        wait->wait_on_signal(_signal);        
    }
    
    if (!from_peer)
    {
        if (_view_base) // if I am a view and not ready, so is my base not ready
            _view_base->get_wait_list(wait, true, false);

        std::unique_lock<std::mutex> lk(_overlap_guard);

        for (const auto &v : _overlap_tensors)
        {
            auto vi = v.lock();
            if (!vi)
                continue;
            vi->get_wait_list(wait, false, true); // can propagate up but not to the base
        }
    }

    // set the signal event on all the other tensors that are my view
    //if the base is not ready, so are all the views
    // only if not coming from a view, so we will not recurse
    if (!from_view) {
        std::unique_lock<std::mutex> lk(_views_guard);     
        for (const auto &v : _views)
        {
            auto vi = v.lock();
            if (!vi)
                continue;
            vi->get_wait_list(wait);
        }       
    }
    return;
}

size_t TensorImpl::get_pos_offset(const Shape &pos) const
{
    size_t ret = _base_offset;
    auto r = rank();
    if (pos.size() != r)
    {
        throw PataThrow(PataRuntimeException() << "Position not in the same rank as tensor.");
    }
    for (unsigned i = 0; i < r; i++)
    {
        ret += pos[i] * _stride[i];
    }
    return ret;
}

void TensorImpl::sync() {   
    if (is_ready()) return;
    auto barrier = BackendManager::Inst().backend()->createBarrierCmd();    
    get_wait_list(barrier);
    
    barrier->wait();
}


bool TensorImpl::is_view_overlap(const std::shared_ptr<TensorImpl> &sibling_view)
{
    auto nrank = rank();
    PATA_ASSERT(nrank == sibling_view->rank());
    PATA_ASSERT(_view_base == sibling_view->_view_base);

    TensorSpace base(_view_base.get());
    TensorSpace lhs(this);
    TensorSpace rhs(sibling_view.get());

    if (_stride != sibling_view->stride())
    { // check overall boundries, hard to predict
        return lhs.is_overlap(rhs);
    }
    // me and him have the same strides
    if (_stride != base.stride)
    {
        // we shall compute the inerleave spfactor per dim
        Shape dilation;
        for (unsigned i = 0; i < nrank; i++)
        {
            size_t f = _stride[i] / base.stride[i];
            PATA_ASSERT(f * base.stride[i] == _stride[i]); // make sure whole integer
            dilation.push_back(f);
        }
        return lhs.is_overlap(rhs, dilation);
    }
    return lhs.is_overlap(rhs);
}

// elementwise add
Tensor Tensor::operator+(const Tensor &rhs)
{
    PATA_ASSERT(shape() == rhs.shape()); // todo : support broadcasting later
    PATA_ASSERT(dtype() == rhs.dtype()); // no implicit casting
    
    auto out_tensor = Tensor(shape(), dtype(), workspace());

    // todo: Can I send my own object? Or maybe copy? does it make sense?
    Add(*this, rhs, out_tensor);
    return out_tensor;
}