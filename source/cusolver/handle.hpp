/*
 * handle.hpp
 *
 *  Created on: 09.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <cusolverDn.h>
#include <memory>

namespace boost {
namespace numeric {
namespace cusolver {

typedef std::shared_ptr<cusolverDnContext> handle_t;

extern const handle_t handle;

}
}
}
