/*
 * amax.cpp
 *
 *  Created on: 26.05.2015
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
struct function<real32_t> { static constexpr auto call = cublasIsamax_v2; };

template <>
struct function<real64_t> { static constexpr auto call = cublasIdamax_v2; };

template <>
struct function<complex32_t> { static constexpr auto call = cublasIcamax_v2; };

template <>
struct function<complex64_t> { static constexpr auto call = cublasIzamax_v2; };

template <typename T>
std::size_t amax(const vector<T>& vector)
{
	typedef typename type_trait<T>::type U;

	int result = 0;

	const cublasStatus_t status = function<T>::call
	(
		handle.get(),
		size(vector),
		reinterpret_cast<const U*>((*vector).get()), 1,
		&result
	);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);

	return result - 1;
}

template std::size_t amax(const vector<real32_t>& vector);
template std::size_t amax(const vector<real64_t>& vector);
template std::size_t amax(const vector<complex32_t>& vector);
template std::size_t amax(const vector<complex64_t>& vector);

}
}
}
