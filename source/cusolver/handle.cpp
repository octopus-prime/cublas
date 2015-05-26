/*
 * handle.cpp
 *
 *  Created on: 09.05.2015
 *      Author: mike_gresens
 */

#include "handle.hpp"
#include "error.hpp"

namespace boost {
namespace numeric {
namespace cusolver {

static cusolverDnHandle_t make_handle()
{
	cusolverDnHandle_t handle = nullptr;
	const cusolverStatus_t status = cusolverDnCreate(&handle);
	if (status != CUSOLVER_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
	return handle;
}

const handle_t handle(make_handle(), cusolverDnDestroy);

}
}
}
