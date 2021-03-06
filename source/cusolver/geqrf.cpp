/*
 * geqrf.cpp
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
void geqrf(cublas::matrix<T>& matrix, cublas::vector<T>& tau, cublas::vector<T>& work, F function)
{
	auto info = cuda::make_container<int>(1);

	const cusolverStatus_t status = function(handle.get(), matrix.rows(), matrix.cols(), (U*) (*matrix).get(), matrix.rows(), (U*) (*tau).get(), (*work).get(), work.size(), info.get());
	if (status != CUSOLVER_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

}

template <>
cublas::vector<real32_t> geqrf(cublas::matrix<real32_t>& matrix, cublas::vector<real32_t>& tau)
{
	auto work = detail::alloc<float>(matrix, cusolverDnSgeqrf_bufferSize);
	detail::geqrf<float>(matrix, tau, work, cusolverDnSgeqrf);
	return std::move(work);
}

template <>
cublas::vector<real64_t> geqrf(cublas::matrix<real64_t>& matrix, cublas::vector<real64_t>& tau)
{
	auto work = detail::alloc<double>(matrix, cusolverDnDgeqrf_bufferSize);
	detail::geqrf<double>(matrix, tau, work, cusolverDnDgeqrf);
	return std::move(work);
}

}
}
}
