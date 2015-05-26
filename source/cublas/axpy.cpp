/*
 * axpy.cpp
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
struct function<real32_t> { static constexpr auto call = cublasSaxpy_v2; };

template <>
struct function<real64_t> { static constexpr auto call = cublasDaxpy_v2; };

template <>
struct function<complex32_t> { static constexpr auto call = cublasCaxpy_v2; };

template <>
struct function<complex64_t> { static constexpr auto call = cublasZaxpy_v2; };

template <typename T, template <typename> class C>
void axpy(const T& alpha, const C<T>& container1, C<T>& container2)
{
	typedef typename type_trait<T>::type U;

	if (size(container1) != size(container2))
		throw std::system_error(CUBLAS_STATUS_INVALID_VALUE, category, __func__);

	const cublasStatus_t status = function<T>::call
	(
		handle.get(),
		size(container2),
		reinterpret_cast<const U*>(&alpha),
		reinterpret_cast<const U*>((*container1).get()), 1,
		reinterpret_cast<U*>((*container2).get()), 1
	);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

template void axpy(const real32_t& alpha, const vector<real32_t>& container1, vector<real32_t>& container2);
template void axpy(const real32_t& alpha, const matrix<real32_t>& container1, matrix<real32_t>& container2);

template void axpy(const real64_t& alpha, const vector<real64_t>& container1, vector<real64_t>& container2);
template void axpy(const real64_t& alpha, const matrix<real64_t>& container1, matrix<real64_t>& container2);

template void axpy(const complex32_t& alpha, const vector<complex32_t>& container1, vector<complex32_t>& container2);
template void axpy(const complex32_t& alpha, const matrix<complex32_t>& container1, matrix<complex32_t>& container2);

template void axpy(const complex64_t& alpha, const vector<complex64_t>& container1, vector<complex64_t>& container2);
template void axpy(const complex64_t& alpha, const matrix<complex64_t>& container1, matrix<complex64_t>& container2);

}
}
}
