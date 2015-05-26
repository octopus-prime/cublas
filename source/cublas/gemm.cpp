/*
 * blas.cpp
 *
 *  Created on: 09.05.2015
 *      Author: mike_gresens
 */

#include <boost/numeric/cublas/blas.hpp>
#include "type_trait.hpp"
#include "size.hpp"
#include "handle.hpp"
#include "error.hpp"

namespace boost {
namespace numeric {
namespace cublas {

template <typename T>
struct function;

template <>
struct function<real32_t> { static constexpr auto call = cublasSgemm_v2; };

template <>
struct function<real64_t> { static constexpr auto call = cublasDgemm_v2; };

template <>
struct function<complex32_t> { static constexpr auto call = cublasCgemm_v2; };

template <>
struct function<complex64_t> { static constexpr auto call = cublasZgemm_v2; };

template <typename T>
void gemm(const T& alpha, const matrix<T>& matrix1, const matrix<T>& matrix2, const T& beta, matrix<T>& matrix3)
{
	typedef typename type_trait<T>::type U;

	const std::size_t m = matrix1.rows();
	const std::size_t n = matrix2.cols();
	const std::size_t k = matrix1.cols();

	if (k != matrix2.rows() || m != matrix3.rows() || n != matrix3.cols())
		throw std::system_error(CUBLAS_STATUS_INVALID_VALUE, category, __func__);

	const cublasStatus_t status = function<T>::call
	(
		handle.get(),
		CUBLAS_OP_N, CUBLAS_OP_N,
		m, n, k,
		reinterpret_cast<const U*>(&alpha),
		reinterpret_cast<const U*>((*matrix1).get()), m,
		reinterpret_cast<const U*>((*matrix2).get()), k,
		reinterpret_cast<const U*>(&beta),
		reinterpret_cast<U*>((*matrix3).get()), m
	);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

template void gemm(const real32_t& alpha, const matrix<real32_t>& matrix1, const matrix<real32_t>& matrix2, const real32_t& beta, matrix<real32_t>& matrix3);
template void gemm(const real64_t& alpha, const matrix<real64_t>& matrix1, const matrix<real64_t>& matrix2, const real64_t& beta, matrix<real64_t>& matrix3);
template void gemm(const complex32_t& alpha, const matrix<complex32_t>& matrix1, const matrix<complex32_t>& matrix2, const complex32_t& beta, matrix<complex32_t>& matrix3);
template void gemm(const complex64_t& alpha, const matrix<complex64_t>& matrix1, const matrix<complex64_t>& matrix2, const complex64_t& beta, matrix<complex64_t>& matrix3);

}
}
}
