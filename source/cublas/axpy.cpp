/*
 * axpy.cpp
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
void axpy(const T& alpha, const cuda::container<T>& container1, cuda::container<T>& container2, F function)
{
	if (container1.size() != container2.size())
		throw std::system_error(CUBLAS_STATUS_INVALID_VALUE, category, __func__);
	const cublasStatus_t status = function(handle.get(), container2.size(), (U*) &alpha, (U*) (*container1).get(), 1, (U*) (*container2).get(), 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

}

template <>
void axpy(const real32_t& alpha, const cuda::container<real32_t>& container1, cuda::container<real32_t>& container2)
{
	detail::axpy<float>(alpha, container1, container2, cublasSaxpy_v2);
}

template <>
void axpy(const real64_t& alpha, const cuda::container<real64_t>& container1, cuda::container<real64_t>& container2)
{
	detail::axpy<double>(alpha, container1, container2, cublasDaxpy_v2);
}

template <>
void axpy(const complex32_t& alpha, const cuda::container<complex32_t>& container1, cuda::container<complex32_t>& container2)
{
	detail::axpy<cuFloatComplex>(alpha, container1, container2, cublasCaxpy_v2);
}

template <>
void axpy(const complex64_t& alpha, const cuda::container<complex64_t>& container1, cuda::container<complex64_t>& container2)
{
	detail::axpy<cuDoubleComplex>(alpha, container1, container2, cublasZaxpy_v2);
}

}
}
}
