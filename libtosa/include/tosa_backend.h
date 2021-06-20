//
// Created by rcohen on 20/6/2021.
//
#pragma once
#ifndef LIBTOSA_TOSA_BACKEND_H
#define LIBTOSA_TOSA_BACKEND_H
#include <string>
#include <memory>
#include <condition_variable>

#include "tosa_stream.h"
#include "tosa_commands.h"

namespace libtosa {
    class Backend {
        public:
        virtual ~Backend() = default;

        virtual Stream *createStream(int id) = 0;

    };

    class BackendManager {
        public:
            typedef enum {
                CPU = 0,
                GAUDI = 1,

                NO_BACKENDS = GAUDI+1
            } BACKEND_TYPE;
            static BackendManager &Inst();            
            void set_backend(BACKEND_TYPE type);
            inline Backend *backend() const {return _backends[_active_backend];}

    private:
        BACKEND_TYPE _active_backend;
        Backend *_backends[NO_BACKENDS];
        BackendManager();
    };

}

#endif // LIBTOSA_TOSA_BACKEND_H
