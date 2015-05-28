/*
 * handle.hpp
 *
 *  Created on: 09.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <cusolverSp.h>
#include <memory>

namespace boost {
namespace numeric {
namespace cusparse {

typedef std::shared_ptr<cusolverSpContext> handle_t;

extern const handle_t handle;

}
}
}
