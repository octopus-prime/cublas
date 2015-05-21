/*
 * handle.hpp
 *
 *  Created on: 09.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <cublas_v2.h>
#include <memory>

namespace boost {
namespace numeric {
namespace cublas {

typedef std::shared_ptr<cublasContext> handle_t;

extern const handle_t handle;

}
}
}
