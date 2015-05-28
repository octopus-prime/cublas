/*
 * csreigvsi.cpp
 *
 *  Created on: 28.05.2015
 *      Author: mike_gresens
 */

#include <boost/numeric/cusparse/sparse.hpp>
#include "type_trait.hpp"
#include "handle.hpp"
#include "descr.hpp"
#include "cusolver/error.hpp"

namespace boost {
namespace numeric {
namespace cusparse {

template <typename T>
struct function;

template <>
struct function<real32_t> { static constexpr auto call = cusolverSpScsreigvsi; };

template <>
struct function<real64_t> { static constexpr auto call = cusolverSpDcsreigvsi; };

template <>
struct function<complex32_t> { static constexpr auto call = cusolverSpCcsreigvsi; };

template <>
struct function<complex64_t> { static constexpr auto call = cusolverSpZcsreigvsi; };

template <typename T>
void csreigvsi(const compressed_matrix<T>& A, const T& mu0, const cublas::vector<T>& x0, const std::size_t max, T& mu, cublas::vector<T>& x)
{
	typedef typename type_trait<T>::type U;

	const cusolverStatus_t status = function<T>::call
	(
		handle.get(),
		A.size(),
		A.nonezero(),
		descr.get(),
		reinterpret_cast<const U*>(A.vals().get()),
		A.rows().get(),
		A.cols().get(),
		reinterpret_cast<const U&>(mu0),
		reinterpret_cast<const U*>((*x0).get()),
		max,
		0, //std::abs(std::numeric_limits<T>::epsilon()),
		reinterpret_cast<U*>(&mu),
		reinterpret_cast<U*>((*x).get())
	);

	if (status != CUSOLVER_STATUS_SUCCESS)
		throw std::system_error(status, cusolver::category, __func__);
}

template void csreigvsi(const compressed_matrix<real32_t>& A, const real32_t& mu0, const cublas::vector<real32_t>& x0, const std::size_t max, real32_t& mu, cublas::vector<real32_t>& x);
template void csreigvsi(const compressed_matrix<real64_t>& A, const real64_t& mu0, const cublas::vector<real64_t>& x0, const std::size_t max, real64_t& mu, cublas::vector<real64_t>& x);
template void csreigvsi(const compressed_matrix<complex32_t>& A, const complex32_t& mu0, const cublas::vector<complex32_t>& x0, const std::size_t max, complex32_t& mu, cublas::vector<complex32_t>& x);
template void csreigvsi(const compressed_matrix<complex64_t>& A, const complex64_t& mu0, const cublas::vector<complex64_t>& x0, const std::size_t max, complex64_t& mu, cublas::vector<complex64_t>& x);

}
}
}
