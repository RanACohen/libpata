//
// Created by rancohen on 15/6/2021.
//
// this file contains several utilities for data manipulation (boilerplate code)
#pragma once

#ifndef LIBTOSA_TOSA_UTILS_H
#define LIBTOSA_TOSA_UTILS_H
#include <list>
#include <memory>

namespace libtosa
{
    static inline size_t __adder(size_t v) { return v; }
    template<typename T, typename... Args>
    static inline size_t __adder(T base, Args... args) {        
        return base + __adder(args...);
    }


    template <typename T>
    struct WeakListReference
    {
        typename std::list<std::weak_ptr<T>> *_list;
        typename std::list<std::weak_ptr<T>>::iterator _item;

        WeakListReference(std::list<std::weak_ptr<T>> *list, const std::shared_ptr<T> &item) : _list(list) 
        {
            _item = list->insert(list->begin(), item);
        }

        ~WeakListReference() { 
            _list->erase(_item); 
            _item = _list->end();
            _list = nullptr;
        }
    };

} // namespace libtosa

#endif