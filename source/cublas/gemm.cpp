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
void gemm(const T& alpha, const matrix<T>& matrix1, const matrix<T>& matrix2, const T& beta, matrix<T>& matrix3, F function)
{
	const std::size_t m = matrix1.rows();
	const std::size_t n = matrix2.cols();
	const std::size_t k = matrix1.cols();
	if (k != matrix2.rows() || m != matrix3.rows() || n != matrix3.cols())
		throw std::system_error(CUBLAS_STATUS_INVALID_VALUE, category, __func__);
	const cublasStatus_t status = function(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, (U*) &alpha, (U*) (*matrix1).get(), m, (U*) (*matrix2).get(), k, (U*) &beta, (U*) (*matrix3).get(), m);
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

}

template <>
void gemm(const real32_t& alpha, const matrix<real32_t>& matrix1, const matrix<real32_t>& matrix2, const real32_t& beta, matrix<real32_t>& matrix3)
{
	detail::gemm<float>(alpha, matrix1, matrix2, beta, matrix3, cublasSgemm_v2);
}

template <>
void gemm(const real64_t& alpha, const matrix<real64_t>& matrix1, const matrix<real64_t>& matrix2, const real64_t& beta, matrix<real64_t>& matrix3)
{
	detail::gemm<double>(alpha, matrix1, matrix2, beta, matrix3, cublasDgemm_v2);
}

template <>
void gemm(const complex32_t& alpha, const matrix<complex32_t>& matrix1, const matrix<complex32_t>& matrix2, const complex32_t& beta, matrix<complex32_t>& matrix3)
{
	detail::gemm<cuFloatComplex>(alpha, matrix1, matrix2, beta, matrix3, cublasCgemm_v2);
}

template <>
void gemm(const complex64_t& alpha, const matrix<complex64_t>& matrix1, const matrix<complex64_t>& matrix2, const complex64_t& beta, matrix<complex64_t>& matrix3)
{
	detail::gemm<cuDoubleComplex>(alpha, matrix1, matrix2, beta, matrix3, cublasZgemm_v2);
}

}
}
}
