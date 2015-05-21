/*
 * lu.cpp
 *
 *  Created on: 19.05.2015
 *      Author: mike_gresens
 */

#include <boost/numeric/cusolver/lu.hpp>
#include <boost/numeric/cusolver/solver.hpp>

namespace boost {
namespace numeric {
namespace cusolver {
namespace lu {

template <typename T>
decomposer<T>::decomposer(const cublas::matrix<T>& matrix)
:
	_matrix(matrix),
	_pivot(std::min(matrix.rows(), matrix.cols()))
{
	getrf(_matrix, _pivot);
}

template <typename T>
decomposer<T>::decomposer(cublas::matrix<T>&& matrix)
:
	_matrix(std::forward<cublas::matrix<T>>(matrix)),
	_pivot(std::min(matrix.rows(), matrix.cols()))
{
	getrf(_matrix, _pivot);
}

template class decomposer<real32_t>;
template class decomposer<real64_t>;
template class decomposer<complex32_t>;
template class decomposer<complex64_t>;

template <typename T>
solver<T>::solver(const cublas::matrix<T>& matrix)
:
	decomposer<T>(matrix)
{
}

template <typename T>
solver<T>::solver(cublas::matrix<T>&& matrix)
:
	decomposer<T>(std::forward<cublas::matrix<T>>(matrix))
{
}

template <typename T>
cublas::vector<T>
solver<T>::operator()(const cublas::vector<T>& vector) const
{
	cublas::vector<T> result(vector);
	getrs(decomposer<T>::_matrix, decomposer<T>::_pivot, result);
	return std::move(result);
}

template class solver<real32_t>;
template class solver<real64_t>;
template class solver<complex32_t>;
template class solver<complex64_t>;

template <typename T>
inverter<T>::inverter(const cublas::matrix<T>& matrix)
:
	decomposer<T>(matrix)
{
}

template <typename T>
inverter<T>::inverter(cublas::matrix<T>&& matrix)
:
	decomposer<T>(std::forward<cublas::matrix<T>>(matrix))
{
}

template <typename T>
cublas::matrix<T>
inverter<T>::operator()() const
{
	const ublas::matrix<T> matrix = ublas::identity_matrix<T>(decomposer<T>::_matrix.rows(), decomposer<T>::_matrix.cols());
	cublas::matrix<T> result(matrix);
	getrs(decomposer<T>::_matrix, decomposer<T>::_pivot, result);
	return std::move(result);
}

template class inverter<real32_t>;
template class inverter<real64_t>;
template class inverter<complex32_t>;
template class inverter<complex64_t>;

}
}
}
}
