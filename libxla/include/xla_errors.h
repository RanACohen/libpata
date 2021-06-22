//
// Created by rancohen on 24/5/2021.
//
#pragma once

#ifndef LIBXLA_XLA_ERRORS_H
#define LIBXLA_XLA_ERRORS_H

#include <exception>
#include <sstream>

namespace libxla {

    class XlaRuntimeException {
        std::stringstream _strstr;
    public:
        XlaRuntimeException() = default;

        template<typename T>
        XlaRuntimeException &operator<<(const T &obj) {
            _strstr << obj;
            return *this;
        }

         std::string msg() const noexcept {
            return _strstr.str();
        }
    };

static inline std::runtime_error XlaThrow(const XlaRuntimeException &ex)
{
    return std::runtime_error(ex.msg());
}

#define XLA_ASSERT(cond) {if (!(cond)) throw XlaThrow(XlaRuntimeException() << #cond);}

}



#endif //LIBXLA_XLA_ERRORS_H
