/*
 * error.cpp
 *
 *  Created on: 10.05.2015
 *      Author: mike_gresens
 */

#include "error.hpp"
#include <cuda_runtime.h>

namespace boost {
namespace numeric {
namespace cuda {

const char*
category_impl::name() const noexcept
{
	return "cuda";
}

std::string
category_impl::message(int status) const
{
	return cudaGetErrorString((cudaError_t) status);
}

const category_impl category(0);

}
}
}
