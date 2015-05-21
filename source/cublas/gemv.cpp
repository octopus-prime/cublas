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
void gemv(const T& alpha, const matrix<T>& matrix, const vector<T>& vector1, const T& beta, vector<T>& vector2, F function)
{
//	if (container1.size() != container2.size())
//		throw std::system_error(CUBLAS_STATUS_INVALID_VALUE, category, __func__);
	const cublasStatus_t status = function(handle.get(), CUBLAS_OP_N, matrix.rows(), matrix.cols(), (U*) &alpha, (U*) (*matrix).get(), matrix.rows(), (U*) (*vector1).get(), 1, (U*) &beta, (U*) (*vector2).get(), 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

}

template <>
void gemv(const real32_t& alpha, const matrix<real32_t>& matrix, const vector<real32_t>& vector1, const real32_t& beta, vector<real32_t>& vector2)
{
	detail::gemv<float>(alpha, matrix, vector1, beta, vector2, cublasSgemv_v2);
}

template <>
void gemv(const real64_t& alpha, const matrix<real64_t>& matrix, const vector<real64_t>& vector1, const real64_t& beta, vector<real64_t>& vector2)
{
	detail::gemv<double>(alpha, matrix, vector1, beta, vector2, cublasDgemv_v2);
}

template <>
void gemv(const complex32_t& alpha, const matrix<complex32_t>& matrix, const vector<complex32_t>& vector1, const complex32_t& beta, vector<complex32_t>& vector2)
{
	detail::gemv<cuFloatComplex>(alpha, matrix, vector1, beta, vector2, cublasCgemv_v2);
}

template <>
void gemv(const complex64_t& alpha, const matrix<complex64_t>& matrix, const vector<complex64_t>& vector1, const complex64_t& beta, vector<complex64_t>& vector2)
{
	detail::gemv<cuDoubleComplex>(alpha, matrix, vector1, beta, vector2, cublasZgemv_v2);
}

}
}
}
