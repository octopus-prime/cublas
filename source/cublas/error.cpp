/*
 * error.cpp
 *
 *  Created on: 10.05.2015
 *      Author: mike_gresens
 */

#include "error.hpp"
#include <cublas_v2.h>
#include <unordered_map>

namespace boost {
namespace numeric {
namespace cublas {

const char*
category_impl::name() const noexcept
{
	return "cublas";
}

std::string
category_impl::message(int status) const
{
	static const std::unordered_map<int, std::string> MAP
	{
		{CUBLAS_STATUS_SUCCESS,				"Success"},
		{CUBLAS_STATUS_NOT_INITIALIZED,		"Not initialized"},
		{CUBLAS_STATUS_ALLOC_FAILED,		"Allocation failed"},
		{CUBLAS_STATUS_INVALID_VALUE,		"Invalid value"},
		{CUBLAS_STATUS_ARCH_MISMATCH,		"Architecture mismatch"},
		{CUBLAS_STATUS_MAPPING_ERROR,		"Mapping error"},
		{CUBLAS_STATUS_EXECUTION_FAILED,	"Execution failed"},
		{CUBLAS_STATUS_INTERNAL_ERROR,		"Internal error"},
		{CUBLAS_STATUS_NOT_SUPPORTED,		"Not supported"},
		{CUBLAS_STATUS_LICENSE_ERROR,		"License error"}
	};
	return MAP.at(status);
}

const category_impl category(0);

}
}
}
