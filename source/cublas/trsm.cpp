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
struct function<real32_t> { static constexpr auto call = cublasStrsm_v2; };

template <>
struct function<real64_t> { static constexpr auto call = cublasDtrsm_v2; };

template <>
struct function<complex32_t> { static constexpr auto call = cublasCtrsm_v2; };

template <>
struct function<complex64_t> { static constexpr auto call = cublasZtrsm_v2; };

template <typename T, template <typename> class C>
void trsm(const T& alpha, const matrix<T>& matrix, C<T>& container)
{
	typedef typename type_trait<T>::type U;

//	if (container1.size() != container2.size())
//		throw std::system_error(CUBLAS_STATUS_INVALID_VALUE, category, __func__);

	const cublasStatus_t status = function<T>::call
	(
		handle.get(),
		CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
		matrix.rows(), cols(container),
		reinterpret_cast<const U*>(&alpha),
		reinterpret_cast<const U*>((*matrix).get()), matrix.rows(),
		reinterpret_cast<U*>((*container).get()), matrix.rows()
	);

	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

template void trsm(const real32_t&, const matrix<real32_t>&, vector<real32_t>&);
template void trsm(const real32_t&, const matrix<real32_t>&, matrix<real32_t>&);

template void trsm(const real64_t&, const matrix<real64_t>&, vector<real64_t>&);
template void trsm(const real64_t&, const matrix<real64_t>&, matrix<real64_t>&);

template void trsm(const complex32_t&, const matrix<complex32_t>&, vector<complex32_t>&);
template void trsm(const complex32_t&, const matrix<complex32_t>&, matrix<complex32_t>&);

template void trsm(const complex64_t&, const matrix<complex64_t>&, vector<complex64_t>&);
template void trsm(const complex64_t&, const matrix<complex64_t>&, matrix<complex64_t>&);

}
}
}
