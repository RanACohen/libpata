//
// Created by rancohen on 23/5/2021.
//

#ifndef LIBTOSA_TOSA_TYPES_H
#define LIBTOSA_TOSA_TYPES_H

namespace libtosa {

    typedef std::vector<int> Shape;
    typedef enum {
        UNKNOWN = 0,
        FLOAT = 1, FLOAT32 = FLOAT, F32 = FLOAT,
        UINT8 = 2, U8 = UINT8,
        INT8 = 3, I8 = INT8,   // int8_t
        UINT16 = 4,  // uint16_t
        INT16 = 5,   // int16_t
        INT32 = 6,   // int32_t
        INT64 = 7,   // int64_t
        STRING = 8,  // string
        BOOL = 9,    // bool
        // IEEE754 half-precision floating-point format (16 bits wide).
        // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
        FLOAT16 = 10,
        DOUBLE = 11, FLOAT64 = DOUBLE, F64 = DOUBLE,
        UINT32 = 12,
        UINT64 = 13,
        COMPLEX64 = 14,     // complex with float32 real and imaginary components
        COMPLEX128 = 15,    // complex with float64 real and imaginary components
        // Non-IEEE floating-point format based on IEEE754 single-precision
        // floating-point number truncated to 16 bits.
        // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
        BFLOAT16 = 16, BF16 = BFLOAT16,
        __LAST = BFLOAT16
    } DType;

    int dtype_byte_size(DType dtype)
    {
        //                            0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16
        static int DTypeByteSize[] = {0, 4, 1, 1, 2, 2, 4, 8, 1, 1, 2, 8, 4, 8, 8,16, 2};
        return ((dtype < 0) || (dtype > DType::__LAST)) ? -1 : DTypeByteSize[dtype];
    }
}

#endif //LIBTOSA_TOSA_TYPES_H
