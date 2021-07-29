//
// Created by rancohen on 15/6/2021.
//
// this file contains several utilities for data manipulation (boilerplate code)
#pragma once

#ifndef LIBPATA_PATA_UTILS_H
#define LIBPATA_PATA_UTILS_H
#include <list>
#include <memory>

namespace libpata
{
    static inline size_t __adder(size_t v) { return v; }
    template<typename T, typename... Args>
    static inline size_t __adder(T base, Args... args) {        
        return base + __adder(args...);
    }


    template <typename T>
    struct WeakListReference
    {
        typename std::list<std::weak_ptr<T>>::iterator _item;
    };

} // namespace libpata

#endif