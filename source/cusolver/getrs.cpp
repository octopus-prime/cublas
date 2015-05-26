/*
 * getrs.cpp
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
void getrs(const cublas::matrix<T>& matrix, const cuda::container<int>& pivot, cublas::vector<T>& vector, F function)
{
	auto info = cuda::make_container<int>(1);

	const cusolverStatus_t status = function(handle.get(), CUBLAS_OP_N, matrix.rows(), 1, (U*) (*matrix).get(), matrix.rows(), pivot.get(), (U*) (*vector).get(), vector.size(), info.get());
	if (status != CUSOLVER_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

template <typename U, typename T, typename F>
void getrs(const cublas::matrix<T>& matrix, const cuda::container<int>& pivot, cublas::matrix<T>& matrix2, F function)
{
	auto info = cuda::make_container<int>(1);

	const cusolverStatus_t status = function(handle.get(), CUBLAS_OP_N, matrix.rows(), matrix2.cols(), (U*) (*matrix).get(), matrix.rows(), pivot.get(), (U*) (*matrix2).get(), matrix.rows(), info.get());
	if (status != CUSOLVER_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

}

template <>
void getrs(const cublas::matrix<real32_t>& matrix, const cuda::container<int>& pivot, cublas::vector<real32_t>& vector)
{
	detail::getrs<float>(matrix, pivot, vector, cusolverDnSgetrs);
}

template <>
void getrs(const cublas::matrix<real64_t>& matrix, const cuda::container<int>& pivot, cublas::vector<real64_t>& vector)
{
	detail::getrs<double>(matrix, pivot, vector, cusolverDnDgetrs);
}

template <>
void getrs(const cublas::matrix<complex32_t>& matrix, const cuda::container<int>& pivot, cublas::vector<complex32_t>& vector)
{
	detail::getrs<cuFloatComplex>(matrix, pivot, vector, cusolverDnCgetrs);
}

template <>
void getrs(const cublas::matrix<complex64_t>& matrix, const cuda::container<int>& pivot, cublas::vector<complex64_t>& vector)
{
	detail::getrs<cuDoubleComplex>(matrix, pivot, vector, cusolverDnZgetrs);
}

template <>
void getrs(const cublas::matrix<real32_t>& matrix, const cuda::container<int>& pivot, cublas::matrix<real32_t>& matrix2)
{
	detail::getrs<float>(matrix, pivot, matrix2, cusolverDnSgetrs);
}

template <>
void getrs(const cublas::matrix<real64_t>& matrix, const cuda::container<int>& pivot, cublas::matrix<real64_t>& matrix2)
{
	detail::getrs<double>(matrix, pivot, matrix2, cusolverDnDgetrs);
}

template <>
void getrs(const cublas::matrix<complex32_t>& matrix, const cuda::container<int>& pivot, cublas::matrix<complex32_t>& matrix2)
{
	detail::getrs<cuFloatComplex>(matrix, pivot, matrix2, cusolverDnCgetrs);
}

template <>
void getrs(const cublas::matrix<complex64_t>& matrix, const cuda::container<int>& pivot, cublas::matrix<complex64_t>& matrix2)
{
	detail::getrs<cuDoubleComplex>(matrix, pivot, matrix2, cusolverDnZgetrs);
}

}
}
}
