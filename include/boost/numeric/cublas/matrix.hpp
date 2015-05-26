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

/**
 * matrix
 */
template <typename T>
class matrix
:
	boost::additive<matrix<T>>,
	boost::multiplicative<matrix<T>, T>
{
public:
	/**
	 * Default constructor.
	 */
	matrix();

	/**
	 * Standard constructor.
	 * @param rows The rows.
	 * @param cols The cols.
	 */
	matrix(const std::size_t rows, const std::size_t cols);

	/**
	 * Convert constructor.
	 * @param matrix The matrix.
	 */
	matrix(const ublas::matrix<T>& matrix);

	/**
	 * Copy constructor.
	 * @param matrix The matrix.
	 */
	matrix(const cublas::matrix<T>& matrix);

//	matrix(cublas::matrix<T>&& matrix) noexcept;

	matrix<T>&
	operator=(const cublas::matrix<T>& matrix);

//	matrix<T>&
//	operator=(cublas::matrix<T>&& matrix) noexcept;

	operator ublas::matrix<T>() const;

	std::size_t
	rows() const noexcept;

	std::size_t
	cols() const noexcept;

	const cuda::container<T>&
	operator*() const noexcept;

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
	cuda::container<T> _elements;
};

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

template <typename T>
const cuda::container<T>&
matrix<T>::operator*() const noexcept
{
	return _elements;
}

template <typename T>
matrix<T>
operator*(const matrix<T>& matrix1, const matrix<T>& matrix2);

}
}
}
