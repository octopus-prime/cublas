/*
 * dot.cpp
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
struct function<real32_t> { static constexpr auto call = cublasSdot_v2; };

template <>
struct function<real64_t> { static constexpr auto call = cublasDdot_v2; };

template <>
struct function<complex32_t> { static constexpr auto call = cublasCdotu_v2; };

template <>
struct function<complex64_t> { static constexpr auto call = cublasZdotu_v2; };

template <typename T>
T dot(const vector<T>& vector1, const vector<T>& vector2)
{
	typedef typename type_trait<T>::type U;

	if (vector1.size() != vector2.size())
		throw std::system_error(CUBLAS_STATUS_INVALID_VALUE, category, __func__);

	T result;

	const cublasStatus_t status = function<T>::call
	(
		handle.get(),
		vector1.size(),
		reinterpret_cast<const U*>((*vector1).get()), 1,
		reinterpret_cast<const U*>((*vector2).get()), 1,
		reinterpret_cast<U*>(&result)
	);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);

	return result;
}

template real32_t dot(const vector<real32_t>& vector1, const vector<real32_t>& vector2);
template real64_t dot(const vector<real64_t>& vector1, const vector<real64_t>& vector2);
template complex32_t dot(const vector<complex32_t>& vector1, const vector<complex32_t>& vector2);
template complex64_t dot(const vector<complex64_t>& vector1, const vector<complex64_t>& vector2);

}
}
}
