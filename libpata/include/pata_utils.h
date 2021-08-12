//
// Created by rancohen on 15/6/2021.
//
// this file contains several utilities for data manipulation (boilerplate code)
#pragma once

#ifndef LIBPATA_PATA_UTILS_H
#define LIBPATA_PATA_UTILS_H
#include <list>
#include <memory>
#include <atomic>
#include <cstring>

#include "pata_errors.h"

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


    template <typename T, int nMaxSize>
    struct SafeGrowOnlyList
    {
        T* list;
        std::atomic_int size;

        typedef T * iterator;
        typedef const T * const_iterator;
        iterator begin() { return &list[0]; }
        iterator end() { return &list[size]; }

        template<typename... TV>
        SafeGrowOnlyList(const TV&... pv)
        {
            list = (T*)malloc(sizeof(T)*nMaxSize);
            memset(list, 0, sizeof(T)*nMaxSize);
            size=0;
            for (auto i : std::initializer_list< std::common_type_t<TV...> >{pv...})
                list[size++] = i;
        }
        
        SafeGrowOnlyList(const SafeGrowOnlyList&src):size((int)src.size)
        {            
            list = (T*)malloc(sizeof(T)*nMaxSize);
            memset(list, 0, sizeof(T)*nMaxSize);
            for (int _i=0; _i<size; _i++)
                list[_i]=src.list[_i];
        }

        SafeGrowOnlyList():size(0){
            list = (T*)malloc(sizeof(T)*nMaxSize);
            memset(list, 0, sizeof(T)*nMaxSize);
        }

        ~SafeGrowOnlyList()
        {
            clear();
            free(list);
        }

        inline T& operator[](unsigned i){ return list[i];}

        void clear()
        {
            for (int _i=0; _i<size; _i++)
                list[_i] = T();
            size=0;
        }

        void add(const T&val)
        {
            int pos = size++;
            PATA_ASSERT (pos < nMaxSize);
            list[pos] = val;
        }

    };

} // namespace libpata

#endif