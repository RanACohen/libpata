//
// Created by rancohen on 24/5/2021.
//
#pragma once

#ifndef LIBTOSA_TOSA_ERRORS_H
#define LIBTOSA_TOSA_ERRORS_H

#include <exception>
#include <sstream>

namespace libtosa {

    class TosaRuntimeException {
        std::stringstream _strstr;
    public:
        TosaRuntimeException() = default;

        template<typename T>
        TosaRuntimeException &operator<<(const T &obj) {
            _strstr << obj;
            return *this;
        }

         std::string msg() const noexcept {
            return _strstr.str();
        }
    };

std::runtime_error TosaThrow(const TosaRuntimeException &ex)
{
    return std::runtime_error(ex.msg());
}

}



#endif //LIBTOSA_TOSA_ERRORS_H
