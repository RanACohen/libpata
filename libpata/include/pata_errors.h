//
// Created by rancohen on 24/5/2021.
//
#pragma once

#ifndef LIBPATA_PATA_ERRORS_H
#define LIBPATA_PATA_ERRORS_H

#include <exception>
#include <sstream>

namespace libpata {

    class PataRuntimeException {
        std::stringstream _strstr;
    public:
        PataRuntimeException() = default;

        template<typename T>
        PataRuntimeException &operator<<(const T &obj) {
            _strstr << obj;
            return *this;
        }

         std::string msg() const noexcept {
            return _strstr.str();
        }
    };

static inline std::runtime_error PataThrow(const PataRuntimeException &ex)
{
    return std::runtime_error(ex.msg());
}

#define PATA_ASSERT(cond) {if (!(cond)) throw PataThrow(PataRuntimeException() << #cond);}

}



#endif //LIBPATA_PATA_ERRORS_H
