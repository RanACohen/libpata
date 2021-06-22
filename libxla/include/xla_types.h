//
// Created by rancohen on 23/5/2021.
//
#pragma once

#ifndef LIBXLA_XLA_TYPES_H
#define LIBXLA_XLA_TYPES_H
#include <vector>

namespace libxla {
    typedef long RelativeSize;

    struct Range {
        RelativeSize start;
        RelativeSize end;
        size_t step;

        // 0,0 means everything
        //Range(int p) { end = p+1; start=p; step=1;} this was an attempt to treat a single int as range of 1
        explicit Range(RelativeSize _end=0) { end = _end; start=0; step=1; }
        explicit Range(RelativeSize _start, RelativeSize _end, unsigned _step=1) { end = _end; start=_start; step=_step; }
    };

    typedef std::vector<size_t> Shape;
    typedef std::vector<Range> TensorRange;

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
        FLOAT16 = 10, FP16 = FLOAT16,
        DOUBLE = 11, FLOAT64 = DOUBLE, F64 = DOUBLE,
        UINT32 = 12,
        UINT64 = 13,
        COMPLEX64 = 14,     // complex with float32 real and imaginary components
        COMPLEX128 = 15,    // complex with float64 real and imaginary components
        // Non-IEEE floating-point format based on IEEE754 single-precision
        // floating-point number truncated to 16 bits.
        // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
        BFLOAT16 = 16, BF16 = BFLOAT16,
        LAST = BFLOAT16
    } DType;

    int dtype_byte_size(DType dtype);


}

#endif //LIBXLA_XLA_TYPES_H
