/*
 * getrf.cpp
 *
 *  Created on: 09.05.2015
 *      Author: mike_gresens
 */

#include <boost/numeric/cusolver/solver.hpp>
#include "handle.hpp"
#include "error.hpp"

namespace boost {
namespace numeric {
namespace cusolver {
namespace detail {

template <typename U, typename T, typename F>
cublas::vector<T> alloc(cublas::matrix<T>& matrix, F function)
{
	int work;

	const cusolverStatus_t status = function(handle.get(), matrix.rows(), matrix.cols(), (U*) (*matrix).get(), matrix.rows(), &work);
	if (status != CUSOLVER_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);

	return cublas::vector<T>(work);
}

template <typename U, typename T, typename F>
void getrf(cublas::matrix<T>& matrix, cuda::container<int>& pivot, cublas::vector<T>& work, F function)
{
	auto info = cuda::make_container<int>(1);

	const cusolverStatus_t status = function(handle.get(), matrix.rows(), matrix.cols(), (U*) (*matrix).get(), matrix.rows(), (U*) (*work).get(), pivot.get(), info.get());
	if (status != CUSOLVER_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

}

template <>
void getrf(cublas::matrix<real32_t>& matrix, cuda::container<int>& pivot)
{
	auto work = detail::alloc<float>(matrix, cusolverDnSgetrf_bufferSize);
	detail::getrf<float>(matrix, pivot, work, cusolverDnSgetrf);
}

template <>
void getrf(cublas::matrix<real64_t>& matrix, cuda::container<int>& pivot)
{
	auto work = detail::alloc<double>(matrix, cusolverDnDgetrf_bufferSize);
	detail::getrf<double>(matrix, pivot, work, cusolverDnDgetrf);
}

template <>
void getrf(cublas::matrix<complex32_t>& matrix, cuda::container<int>& pivot)
{
	auto work = detail::alloc<cuFloatComplex>(matrix, cusolverDnCgetrf_bufferSize);
	detail::getrf<cuFloatComplex>(matrix, pivot, work, cusolverDnCgetrf);
}

template <>
void getrf(cublas::matrix<complex64_t>& matrix, cuda::container<int>& pivot)
{
	auto work = detail::alloc<cuDoubleComplex>(matrix, cusolverDnZgetrf_bufferSize);
	detail::getrf<cuDoubleComplex>(matrix, pivot, work, cusolverDnZgetrf);
}

}
}
}
