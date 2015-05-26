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
struct function<real32_t> { static constexpr auto call = cublasSgemv_v2; };

template <>
struct function<real64_t> { static constexpr auto call = cublasDgemv_v2; };

template <>
struct function<complex32_t> { static constexpr auto call = cublasCgemv_v2; };

template <>
struct function<complex64_t> { static constexpr auto call = cublasZgemv_v2; };

template <typename T>
void gemv(const T& alpha, const matrix<T>& matrix, const vector<T>& vector1, const T& beta, vector<T>& vector2)
{
	typedef typename type_trait<T>::type U;

//	if (container1.size() != container2.size())
//		throw std::system_error(CUBLAS_STATUS_INVALID_VALUE, category, __func__);

	const cublasStatus_t status = function<T>::call
	(
		handle.get(),
		CUBLAS_OP_N,
		matrix.rows(), matrix.cols(),
		reinterpret_cast<const U*>(&alpha),
		reinterpret_cast<const U*>((*matrix).get()), matrix.rows(),
		reinterpret_cast<const U*>((*vector1).get()), 1,
		reinterpret_cast<const U*>(&beta),
		reinterpret_cast<U*>((*vector2).get()), 1
	);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

template void gemv(const real32_t& alpha, const matrix<real32_t>& matrix, const vector<real32_t>& vector1, const real32_t& beta, vector<real32_t>& vector2);
template void gemv(const real64_t& alpha, const matrix<real64_t>& matrix, const vector<real64_t>& vector1, const real64_t& beta, vector<real64_t>& vector2);
template void gemv(const complex32_t& alpha, const matrix<complex32_t>& matrix, const vector<complex32_t>& vector1, const complex32_t& beta, vector<complex32_t>& vector2);
template void gemv(const complex64_t& alpha, const matrix<complex64_t>& matrix, const vector<complex64_t>& vector1, const complex64_t& beta, vector<complex64_t>& vector2);

}
}
}
