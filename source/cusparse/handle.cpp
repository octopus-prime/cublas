/*
 * handle.cpp
 *
 *  Created on: 09.05.2015
 *      Author: mike_gresens
 */

#include "handle.hpp"
#include "cusolver/error.hpp"

namespace boost {
namespace numeric {
namespace cusparse {

static cusolverSpHandle_t make_handle()
{
	cusolverSpHandle_t handle = nullptr;
	const cusolverStatus_t status = cusolverSpCreate(&handle);
	if (status != CUSOLVER_STATUS_SUCCESS)
		throw std::system_error(status, cusolver::category, __func__);
	return handle;
}

const handle_t handle(make_handle(), cusolverSpDestroy);

}
}
}
