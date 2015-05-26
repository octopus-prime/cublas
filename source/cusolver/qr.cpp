/*
 * qr.cpp
 *
 *  Created on: 19.05.2015
 *      Author: mike_gresens
 */

#include <boost/numeric/cusolver/qr.hpp>
#include <boost/numeric/cusolver/solver.hpp>
#include <boost/numeric/cublas/blas.hpp>

namespace boost {
namespace numeric {
namespace cusolver {
namespace qr {

template <typename T>
decomposer<T>::decomposer(const cublas::matrix<T>& matrix)
:
	_matrix(matrix),
	_tau(std::min(matrix.rows(), matrix.cols())),
	_work(geqrf(_matrix, _tau))
{
}

template class decomposer<real32_t>;
template class decomposer<real64_t>;
//template class decomposer<complex32_t>;
//template class decomposer<complex64_t>;

template <typename T>
solver<T>::solver(const cublas::matrix<T>& matrix)
:
	decomposer<T>(matrix)
{
}

template <typename T>
cublas::vector<T>
solver<T>::operator()(const cublas::vector<T>& vector) const
{
	cublas::vector<T> result(vector);
	ormqr(decomposer<T>::_matrix, decomposer<T>::_tau, const_cast<cublas::vector<T>&>(decomposer<T>::_work), result);
	cublas::trsm(T(1), decomposer<T>::_matrix, result);
	return std::move(result);
}

template class solver<real32_t>;
template class solver<real64_t>;
//template class solver<complex32_t>;
//template class solver<complex64_t>;

template <typename T>
inverter<T>::inverter(const cublas::matrix<T>& matrix)
:
	decomposer<T>(matrix)
{
}

template <typename T>
cublas::matrix<T>
inverter<T>::operator()() const
{
	const ublas::matrix<T> matrix = ublas::identity_matrix<T>(decomposer<T>::_matrix.rows(), decomposer<T>::_matrix.cols());
	cublas::matrix<T> result(matrix);
	ormqr(decomposer<T>::_matrix, decomposer<T>::_tau, const_cast<cublas::vector<T>&>(decomposer<T>::_work), result);
	cublas::trsm(T(1), decomposer<T>::_matrix, result);
	return std::move(result);
}

template class inverter<real32_t>;
template class inverter<real64_t>;
//template class inverter<complex32_t>;
//template class inverter<complex64_t>;

}
}
}
}
