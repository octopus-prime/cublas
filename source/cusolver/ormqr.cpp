/*
 * ormqr.cpp
 *
 *  Created on: 19.05.2015
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
void ormqr(const cublas::matrix<T>& matrix, const cublas::vector<T>& tau, cublas::vector<T>& work, cublas::vector<T>& vector, F function)
{
	auto info = cuda::make_container<int>(1);

	const cusolverStatus_t status = function(handle.get(), CUBLAS_SIDE_LEFT, CUBLAS_OP_T, matrix.rows(), 1, matrix.cols(), (U*) (*matrix).get(), matrix.rows(), (*tau).get(), (U*) (*vector).get(), matrix.rows(), (*work).get(), work.size(), info.get());
	if (status != CUSOLVER_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

template <typename U, typename T, typename F>
void ormqr(const cublas::matrix<T>& matrix, const cublas::vector<T>& tau, cublas::vector<T>& work, cublas::matrix<T>& matrix2, F function)
{
	auto info = cuda::make_container<int>(1);

	const cusolverStatus_t status = function(handle.get(), CUBLAS_SIDE_LEFT, CUBLAS_OP_T, matrix.rows(), matrix2.cols(), matrix.cols(), (U*) (*matrix).get(), matrix.rows(), (*tau).get(), (U*) (*matrix2).get(), matrix.rows(), (*work).get(), work.size(), info.get());
	if (status != CUSOLVER_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

}

template <>
void ormqr(const cublas::matrix<real32_t>& matrix, const cublas::vector<real32_t>& tau, cublas::vector<real32_t>& work, cublas::vector<real32_t>& vector)
{
	detail::ormqr<float>(matrix, tau, work, vector, cusolverDnSormqr);
}

template <>
void ormqr(const cublas::matrix<real64_t>& matrix, const cublas::vector<real64_t>& tau, cublas::vector<real64_t>& work, cublas::vector<real64_t>& vector)
{
	detail::ormqr<double>(matrix, tau, work, vector, cusolverDnDormqr);
}

template <>
void ormqr(const cublas::matrix<real32_t>& matrix, const cublas::vector<real32_t>& tau, cublas::vector<real32_t>& work, cublas::matrix<real32_t>& matrix2)
{
	detail::ormqr<float>(matrix, tau, work, matrix2, cusolverDnSormqr);
}

template <>
void ormqr(const cublas::matrix<real64_t>& matrix, const cublas::vector<real64_t>& tau, cublas::vector<real64_t>& work, cublas::matrix<real64_t>& matrix2)
{
	detail::ormqr<double>(matrix, tau, work, matrix2, cusolverDnDormqr);
}

}
}
}
