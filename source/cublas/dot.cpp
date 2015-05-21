/*
 * dot.cpp
 *
 *  Created on: 09.05.2015
 *      Author: mike_gresens
 */

#include <boost/numeric/cublas/blas.hpp>
#include "handle.hpp"
#include "error.hpp"

namespace boost {
namespace numeric {
namespace cublas {
namespace detail {

template <typename U, typename T, typename F>
T dot(const vector<T>& vector1, const vector<T>& vector2, F function)
{
	if (vector1.size() != vector2.size())
		throw std::system_error(CUBLAS_STATUS_INVALID_VALUE, category, __func__);
	T result;
	const cublasStatus_t status = function(handle.get(), vector1.size(), (U*) (*vector1).get(), 1, (U*) (*vector2).get(), 1, (U*) &result);
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
	return result;
}

}

template <>
real32_t dot(const vector<real32_t>& vector1, const vector<real32_t>& vector2)
{
	return detail::dot<float>(vector1, vector2, cublasSdot_v2);
}

template <>
real64_t dot(const vector<real64_t>& vector1, const vector<real64_t>& vector2)
{
	return detail::dot<double>(vector1, vector2, cublasDdot_v2);
}

template <>
complex32_t dot(const vector<complex32_t>& vector1, const vector<complex32_t>& vector2)
{
	return detail::dot<cuFloatComplex>(vector1, vector2, cublasCdotu_v2);
}

template <>
complex64_t dot(const vector<complex64_t>& vector1, const vector<complex64_t>& vector2)
{
	return detail::dot<cuDoubleComplex>(vector1, vector2, cublasZdotu_v2);
}

}
}
}
