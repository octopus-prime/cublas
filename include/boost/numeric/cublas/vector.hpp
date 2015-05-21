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
	public cuda::container<T>,
	boost::additive<vector<T>>,
	boost::multiplicative<vector<T>, T>
{
public:
	vector(const std::size_t size);

	vector(const ublas::vector<T>& vector);

	operator ublas::vector<T>() const;

	vector<T>&
	operator+=(const vector<T>& vector);

	vector<T>&
	operator-=(const vector<T>& vector);

	vector<T>&
	operator*=(const T& value);

	vector<T>&
	operator/=(const T& value);
};

}
}
}
