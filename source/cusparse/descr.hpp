/*
 * descr.hpp
 *
 *  Created on: 28.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <cusolverSp.h>
#include <memory>

namespace boost {
namespace numeric {
namespace cusparse {

typedef std::shared_ptr<cusparseMatDescr> descr_t;

extern const descr_t descr;

}
}
}
