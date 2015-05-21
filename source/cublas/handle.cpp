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
namespace cublas {

static cublasHandle_t make_handle()
{
	cublasHandle_t handle;
	const cublasStatus_t status = cublasCreate_v2(&handle);
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
	return handle;
}

const handle_t handle(make_handle(), cublasDestroy_v2);

}
}
}
