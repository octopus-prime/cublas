/*
 * scal.cpp
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
void scal(const T& alpha, cuda::container<T>& container, F function)
{
	const cublasStatus_t status = function(handle.get(), container.size(), (U*) &alpha, (U*) (*container).get(), 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

}

template <>
void scal(const real32_t& alpha, cuda::container<real32_t>& container)
{
	detail::scal<float>(alpha, container, cublasSscal_v2);
}

template <>
void scal(const real64_t& alpha, cuda::container<real64_t>& container)
{
	detail::scal<double>(alpha, container, cublasDscal_v2);
}

template <>
void scal(const complex32_t& alpha, cuda::container<complex32_t>& container)
{
	detail::scal<cuFloatComplex>(alpha, container, cublasCscal_v2);
}

template <>
void scal(const complex64_t& alpha, cuda::container<complex64_t>& container)
{
	detail::scal<cuDoubleComplex>(alpha, container, cublasZscal_v2);
}

}
}
}
