/*
 * error.cpp
 *
 *  Created on: 10.05.2015
 *      Author: mike_gresens
 */

#include "error.hpp"
#include <cusolverDn.h>
#include <unordered_map>

namespace boost {
namespace numeric {
namespace cusolver {

const char*
category_impl::name() const noexcept
{
	return "cusolver";
}

std::string
category_impl::message(int status) const
{
	static const std::unordered_map<int, std::string> MAP
	{
		{CUSOLVER_STATUS_SUCCESS,					"Success"},
		{CUSOLVER_STATUS_NOT_INITIALIZED,			"Not initialized"},
		{CUSOLVER_STATUS_ALLOC_FAILED,				"Allocation failed"},
		{CUSOLVER_STATUS_INVALID_VALUE,				"Invalid value"},
		{CUSOLVER_STATUS_ARCH_MISMATCH,				"Architecture mismatch"},
		{CUSOLVER_STATUS_MAPPING_ERROR,				"Mapping error"},
		{CUSOLVER_STATUS_EXECUTION_FAILED,			"Execution failed"},
		{CUSOLVER_STATUS_INTERNAL_ERROR,			"Internal error"},
		{CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED,	"Matrix type not supported"},
		{CUSOLVER_STATUS_NOT_SUPPORTED,				"Not supported"},
		{CUSOLVER_STATUS_ZERO_PIVOT,				"Zero pivot"},
		{CUSOLVER_STATUS_INVALID_LICENSE,			"License error"}
	};
	return MAP.at(status);
}

const category_impl category(0);

}
}
}
