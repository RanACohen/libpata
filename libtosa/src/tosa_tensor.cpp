#include "tosa_tensor.h"

using namespace libtosa;

int dtype_byte_size(DType dtype)
{
    //                            0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
    static int DTypeByteSize[] = {0, 4, 1, 1, 2, 2, 4, 8, 1, 1, 2, 8, 4, 8, 8,16, 2};
    return ((dtype < 0) || (dtype > DType::LAST)) ? -1 : DTypeByteSize[dtype];
}



Tensor::Tensor(const Shape &shape, DType dtype) {
    _dtype = dtype;
    _shape = shape;
    int s=1;
    _stride.push_back(1);
    for (unsigned i=shape.size()-1; i>0; --i)
    {
        s *= shape[i];
        _stride.insert(_stride.begin(), s);
    }
}

Tensor::Tensor(const Shape &shape, const Shape &stride, DType dtype) {
    _dtype = dtype;
    _shape = shape;
    _stride = stride;
}

Tensor::Tensor(const TensorPtr &base, const TensorRange &range) {
    _dtype = base->_dtype;
    
}
