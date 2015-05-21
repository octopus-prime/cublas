/*
 * lu.hpp
 *
 *  Created on: 19.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <boost/numeric/cublas/matrix.hpp>
#include <boost/numeric/cublas/vector.hpp>

namespace boost {
namespace numeric {
namespace cusolver {
namespace lu {

template <typename T>
class decomposer
{
public:
	decomposer(const cublas::matrix<T>& matrix);
	decomposer(cublas::matrix<T>&& matrix);
	operator const cublas::matrix<T>&() const noexcept;

protected:
	cublas::matrix<T> _matrix;
	cuda::container<int> _pivot;
};

template <typename T>
decomposer<T>::operator const cublas::matrix<T>&() const noexcept
{
	return _matrix;
}

template <typename T>
class solver
:
	public decomposer<T>
{
public:
	solver(const cublas::matrix<T>& matrix);
	solver(cublas::matrix<T>&& matrix);

	cublas::vector<T>
	operator()(const cublas::vector<T>& vector) const;
};

template <typename T>
class inverter
:
	public decomposer<T>
{
public:
	inverter(const cublas::matrix<T>& matrix);
	inverter(cublas::matrix<T>&& matrix);

	cublas::matrix<T>
	operator()() const;
};

}
}
}
}
