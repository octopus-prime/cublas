/*
 * csrlsvqr.cpp
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
struct function<real32_t> { static constexpr auto call = cusolverSpScsrlsvqr; };

template <>
struct function<real64_t> { static constexpr auto call = cusolverSpDcsrlsvqr; };

template <>
struct function<complex32_t> { static constexpr auto call = cusolverSpCcsrlsvqr; };

template <>
struct function<complex64_t> { static constexpr auto call = cusolverSpZcsrlsvqr; };

template <typename T>
void csrlsvqr(const compressed_matrix<T>& A, const cublas::vector<T>& b, cublas::vector<T>& x)
{
	typedef typename type_trait<T>::type U;

	int singularity = 0;
	const cusolverStatus_t status = function<T>::call
	(
		handle.get(),
		A.size(),
		A.nonezero(),
		descr.get(),
		reinterpret_cast<const U*>(A.vals().get()),
		A.rows().get(),
		A.cols().get(),
		reinterpret_cast<const U*>((*b).get()),
		0, //std::abs(std::numeric_limits<T>::epsilon()),
		0,
		reinterpret_cast<U*>((*x).get()),
		&singularity
	);

	if (status != CUSOLVER_STATUS_SUCCESS)
		throw std::system_error(status, cusolver::category, __func__);
}

template void csrlsvqr(const compressed_matrix<real32_t>& A, const cublas::vector<real32_t>& b, cublas::vector<real32_t>& x);
template void csrlsvqr(const compressed_matrix<real64_t>& A, const cublas::vector<real64_t>& b, cublas::vector<real64_t>& x);
template void csrlsvqr(const compressed_matrix<complex32_t>& A, const cublas::vector<complex32_t>& b, cublas::vector<complex32_t>& x);
template void csrlsvqr(const compressed_matrix<complex64_t>& A, const cublas::vector<complex64_t>& b, cublas::vector<complex64_t>& x);

}
}
}
