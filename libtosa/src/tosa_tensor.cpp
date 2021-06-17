#include <iostream>

#include "tosa_tensor.h"
#include "tosa_errors.h"
#include "tosa_commands.h"
#include "tosa_operator.h"

using namespace libtosa;

int libtosa::dtype_byte_size(DType dtype)
{
    //                            0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    static int DTypeByteSize[] = {0, 4, 1, 1, 2, 2, 4, 8, 1, 1, 2, 8, 4, 8, 8, 16, 2};
    return ((dtype < 0) || (dtype > DType::LAST)) ? -1 : DTypeByteSize[dtype];
}


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
    s *= shape[0] * dtype_byte_size(dtype);
    _memory = MemoryBlock::allocate(s, workspace);
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

    for (unsigned i = 0; i < base->rank(); i++)
    {
        auto range = i < t_range.size() ? t_range[i] : Range();
        // handle negative like in python: if end ==0 means the end as -1 is one before (unlike start).
        auto end = range.end > 0 ? range.end : base_shape[i] + range.end;
        auto start = range.start >= 0 ? range.start : base_shape[i] + range.start;
        // make sure no out of bound
        CLIP(end, 0, base_shape[i]);
        CLIP(start, 0, base_shape[i]);
        _shape.push_back((end - start + range.step - 1) / range.step); // ceil: step 7 in 22 :4=ciel(22/7) 0,7,14,21
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
   for (auto &ot : _overlap_tensors)
   {
        TensorPtr t = ot.lock();
        if (t)
            t->remove_overlap(this);
        else {
            std::cout << "oops!\n";
        }
   }
}

void TensorImpl::remove_overlap(TensorImpl *peer)
{
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
    // todo: add a backend!
    set_signal(std::make_shared<CPUSignal>());
}

CommandPtr TensorImpl::getWaitIfNotReady()
{
    //todo: move moutext to wait signal
    std::lock_guard<std::mutex> guard(_signal_mutex);
    if (_signal && !_signal->is_ready())
    {
        return _signal->getWaitCmd();
    }
    return CommandPtr();
}

void TensorImpl::set_signal(const std::shared_ptr<Signal> &signal, bool from_view, bool from_peer)
{
    TOSA_ASSERT(!_signal); // if this tensor is not ready we cannot overide it
    _signal = signal;

    if (!from_peer)
    {
        if (_view_base) // if I am a view and not ready, so is my base not ready
            _view_base->set_signal(signal, true, false);

        for (const auto &v : _overlap_tensors)
        {
            auto vi = v.lock();
            if (!vi)
                continue;
            vi->set_signal(signal, false, true); // can propaget up but no do the base
        }
    }

    // set the signal event on all the other tensors that are my view
    //if the base is not ready, so are all the views
    // only if not coming from a view, so we will not recurse
    if (from_view)
        return;
    for (const auto &v : _views)
    {
        auto vi = v.lock();
        if (!vi)
            continue;
        vi->set_signal(signal);
    }
}

size_t TensorImpl::get_pos_offset(const Shape &pos)
{
    size_t ret = _base_offset;
    auto r = rank();
    if (pos.size() != r)
    {
        throw TosaThrow(TosaRuntimeException() << "Position not in the same rank as tensor.");
    }
    for (unsigned i = 0; i < r; i++)
    {
        ret += pos[i] * _stride[i];
    }
    return ret;
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

void TensorImpl::register_as_view(const TensorPtr &view)
{
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
            vi->_overlap_tensors.push_back(view);
            view->_overlap_tensors.push_back(vi);            
        }
    }
    view->add_me_to(_views);
}

bool TensorImpl::is_view_overlap(const std::shared_ptr<TensorImpl> &sibling_view)
{
    auto nrank = rank();
    TOSA_ASSERT(nrank == sibling_view->rank());
    TOSA_ASSERT(_view_base == sibling_view->_view_base);

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
            TOSA_ASSERT(f * base.stride[i] == _stride[i]); // make sure whole integer
            dilation.push_back(f);
        }
        return lhs.is_overlap(rhs, dilation);
    }
    return lhs.is_overlap(rhs);
}

// elementwise add
Tensor Tensor::operator+(const Tensor &rhs) const
{
    TOSA_ASSERT(shape() == rhs.shape()); // todo : support broadcasting later
    TOSA_ASSERT(dtype() == rhs.dtype()); // no implicit casting

    AttrList attributes;
    auto out_tensor = Tensor(shape(), dtype(), workspace());

    // todo: Can I send my own object? Or maybe copy? does it make sense?
    schedule("tosa.add", {*this, rhs}, {out_tensor}, attributes);
    return out_tensor;
}