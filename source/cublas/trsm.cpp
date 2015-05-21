/*
 * blas.cpp
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
void trsm(const T& alpha, const matrix<T>& matrix, vector<T>& vector, F function)
{
//	if (container1.size() != container2.size())
//		throw std::system_error(CUBLAS_STATUS_INVALID_VALUE, category, __func__);
	const cublasStatus_t status = function(handle.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, matrix.rows(), 1, (U*) &alpha, (U*) (*matrix).get(), matrix.rows(), (U*) (*vector).get(), matrix.rows());
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

template <typename U, typename T, typename F>
void trsm(const T& alpha, const matrix<T>& matrix1, matrix<T>& matrix2, F function)
{
//	if (container1.size() != container2.size())
//		throw std::system_error(CUBLAS_STATUS_INVALID_VALUE, category, __func__);
	const cublasStatus_t status = function(handle.get(), CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, matrix1.rows(), matrix2.cols(), (U*) &alpha, (U*) (*matrix1).get(), matrix1.rows(), (U*) (*matrix2).get(), matrix1.rows());
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

}

template <>
void trsm(const real32_t& alpha, const matrix<real32_t>& matrix, vector<real32_t>& vector)
{
	detail::trsm<float>(alpha, matrix, vector, cublasStrsm_v2);
}

template <>
void trsm(const real64_t& alpha, const matrix<real64_t>& matrix, vector<real64_t>& vector)
{
	detail::trsm<double>(alpha, matrix, vector, cublasDtrsm_v2);
}

template <>
void trsm(const complex32_t& alpha, const matrix<complex32_t>& matrix, vector<complex32_t>& vector)
{
	detail::trsm<cuFloatComplex>(alpha, matrix, vector, cublasCtrsm_v2);
}

template <>
void trsm(const complex64_t& alpha, const matrix<complex64_t>& matrix, vector<complex64_t>& vector)
{
	detail::trsm<cuDoubleComplex>(alpha, matrix, vector, cublasZtrsm_v2);
}

template <>
void trsm(const real32_t& alpha, const matrix<real32_t>& matrix1, matrix<real32_t>& matrix2)
{
	detail::trsm<float>(alpha, matrix1, matrix2, cublasStrsm_v2);
}

template <>
void trsm(const real64_t& alpha, const matrix<real64_t>& matrix1, matrix<real64_t>& matrix2)
{
	detail::trsm<double>(alpha, matrix1, matrix2, cublasDtrsm_v2);
}

template <>
void trsm(const complex32_t& alpha, const matrix<complex32_t>& matrix1, matrix<complex32_t>& matrix2)
{
	detail::trsm<cuFloatComplex>(alpha, matrix1, matrix2, cublasCtrsm_v2);
}

template <>
void trsm(const complex64_t& alpha, const matrix<complex64_t>& matrix1, matrix<complex64_t>& matrix2)
{
	detail::trsm<cuDoubleComplex>(alpha, matrix1, matrix2, cublasZtrsm_v2);
}

}
}
}
