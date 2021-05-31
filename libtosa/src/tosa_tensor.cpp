#include "tosa_tensor.h"
#include "tosa_errors.h"

using namespace libtosa;

int libtosa::dtype_byte_size(DType dtype)
{
    //                            0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    static int DTypeByteSize[] = {0, 4, 1, 1, 2, 2, 4, 8, 1, 1, 2, 8, 4, 8, 8,16, 2};
    return ((dtype < 0) || (dtype > DType::LAST)) ? -1 : DTypeByteSize[dtype];
}

typedef std::shared_ptr<TensorImpl> TensorPtr;


TensorImpl::TensorImpl(const Shape &shape, DType dtype, const WorkspacePtr &workspace) {
    _dtype = dtype;
    _shape = shape;
    size_t s=1;
    _stride.push_back(1);
    for (unsigned i=shape.size()-1; i>0; --i)
    {
        s *= shape[i];
        _stride.insert(_stride.begin(), s);
    }
    s *= shape[0]*dtype_byte_size(dtype);
    _memory = MemoryBlock::allocate(s, workspace);
}

/**
 * Allocate with custom streides - this is the developer responsibility, should not be used really.
 * use Allocator with Range for custom strides
 * @param shape
 * @param stride
 * @param dtype
 * @param workspace
 */
TensorImpl::TensorImpl(const Shape &shape, const Shape &stride, DType dtype, const WorkspacePtr &workspace) {
    _dtype = dtype;
    _shape = shape;
    _stride = stride;

    auto s = stride[0]*shape[0]*dtype_byte_size(dtype);
    _memory = MemoryBlock::allocate(s, workspace);
}

#define CLIP(v,s,e) v = v < (s) ? (s) : v > (e) ? (e) : v

TensorImpl::TensorImpl(const TensorPtr &base, const TensorRange &t_range) {
    _dtype = base->_dtype;
    _view_base = base;
    auto base_shape = base->shape();
    _stride = base->stride();
    _memory = base->_memory;
    Shape start_pos;
    for(unsigned i=0; i<base->rank(); i++)
    {
        auto range = i<t_range.size() ? t_range[i] : Range();
        // handle negative like in python: if end ==0 means the end as -1 is one before (unlike start).
        auto end = range.end > 0 ? range.end : base_shape[i]+range.end;
        auto start = range.start >= 0 ? range.start : base_shape[i]+range.start;
        // make sure no out of bound
        CLIP(end, 0, base_shape[i]);
        CLIP(start, 0, base_shape[i]);
        _shape.push_back((end-start+range.step-1)/range.step); // ceil: step 7 in 22 :4=ciel(22/7) 0,7,14,21
        if (range.step != 1)
        {
            _stride[i] *= range.step;
        }
        start_pos.push_back(start);
    }
    _base_offset = base->get_pos_offset(start_pos);
}


size_t TensorImpl::get_pos_offset(const Shape &pos) {
    size_t ret = _base_offset;
    auto r = rank();
    if (pos.size() != r)
    {
        throw TosaThrow(TosaRuntimeException() << "Position not in the same rank as tensor.");
    }
    for (unsigned i=0; i<r; i++)
    {
        ret += pos[i]*_stride[i];
    }
    return ret;
}

Tensor Tensor::operator+(const Tensor &rhs) const {
    TOSA_ASSERT(shape() == rhs.shape()); // todo : support broadcasting later
    TOSA_ASSERT(dtype() == rhs.dtype()); // no implicit casting
    // allocate output of add in the same workspace as the left hand side
    auto out_tensor = Tensor(shape(), dtype(), workspace());
    // todo: Gal - impl me

    return out_tensor;
}
