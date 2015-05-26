/*
 * scal.cpp
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
struct function<real32_t>
{
	static constexpr auto call = cublasSscal_v2;
};

template <>
struct function<real64_t>
{
	static constexpr auto call = cublasDscal_v2;
};

template <>
struct function<complex32_t>
{
	static constexpr auto call = cublasCscal_v2;
};

template <>
struct function<complex64_t>
{
	static constexpr auto call = cublasZscal_v2;
};

template <typename T, template <typename> class C>
void scal(const T& alpha, C<T>& container)
{
	typedef typename type_trait<T>::type U;

	const cublasStatus_t status = function<T>::call
	(
		handle.get(),
		size(container),
		reinterpret_cast<const U*>(&alpha),
		reinterpret_cast<U*>((*container).get()), 1
	);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

template void scal(const real32_t& alpha, vector<real32_t>& container);
template void scal(const real32_t& alpha, matrix<real32_t>& container);

template void scal(const real64_t& alpha, vector<real64_t>& container);
template void scal(const real64_t& alpha, matrix<real64_t>& container);

template void scal(const complex32_t& alpha, vector<complex32_t>& container);
template void scal(const complex32_t& alpha, matrix<complex32_t>& container);

template void scal(const complex64_t& alpha, vector<complex64_t>& container);
template void scal(const complex64_t& alpha, matrix<complex64_t>& container);

}
}
}
