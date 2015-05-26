/*
 * vector.hpp
 *
 *  Created on: 13.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <boost/numeric/cuda/container.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/operators.hpp>

namespace boost {
namespace numeric {
namespace cublas {

template <typename T>
class vector
:
	boost::additive<vector<T>>,
	boost::multiplicative<vector<T>, T>
{
public:
	vector();

	vector(const std::size_t size);

	vector(const ublas::vector<T>& vector);

	vector(const cublas::vector<T>& vector);

//	vector(cublas::vector<T>&& vector);

	vector<T>&
	operator=(const cublas::vector<T>& vector);

//	vector<T>&
//	operator=(cublas::vector<T>&& vector) noexcept;

	operator ublas::vector<T>() const;

	std::size_t
	size() const noexcept;

	const cuda::container<T>&
	operator*() const noexcept;

	vector<T>&
	operator+=(const vector<T>& vector);

	vector<T>&
	operator-=(const vector<T>& vector);

	vector<T>&
	operator*=(const T& value);

	vector<T>&
	operator/=(const T& value);

private:
	std::size_t _size;
	cuda::container<T> _elements;
};

template <typename T>
std::size_t
vector<T>::size() const noexcept
{
	return _size;
}

template <typename T>
const cuda::container<T>&
vector<T>::operator*() const noexcept
{
	return _elements;
}

template <typename T>
T
operator*(const vector<T>& vector1, const vector<T>& vector2);

}
}
}
