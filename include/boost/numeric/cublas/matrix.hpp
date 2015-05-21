/*
 * matrix.hpp
 *
 *  Created on: 14.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <boost/numeric/cuda/container.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/operators.hpp>

namespace boost {
namespace numeric {
namespace cublas {

template <typename T>
class matrix
:
	public cuda::container<T>,
	boost::additive<matrix<T>>,
	boost::multiplicative<matrix<T>, T>
{
public:
	matrix(const std::size_t rows, const std::size_t cols);

	matrix(const ublas::matrix<T>& matrix);

	operator ublas::matrix<T>() const;

	std::size_t
	rows() const noexcept;

	std::size_t
	cols() const noexcept;

	matrix<T>&
	operator+=(const matrix<T>& matrix);

	matrix<T>&
	operator-=(const matrix<T>& matrix);

	matrix<T>&
	operator*=(const T& value);

	matrix<T>&
	operator/=(const T& value);

private:
	std::size_t _rows;
	std::size_t _cols;
};

template <typename T>
matrix<T>
operator*(const matrix<T>& matrix1, const matrix<T>& matrix2);

template <typename T>
std::size_t
matrix<T>::rows() const noexcept
{
	return _rows;
}

template <typename T>
std::size_t
matrix<T>::cols() const noexcept
{
	return _cols;
}

}
}
}
