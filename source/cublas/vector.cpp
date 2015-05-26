/*
 * vector.cpp
 *
 *  Created on: 14.05.2015
 *      Author: mike_gresens
 */

#include <boost/numeric/cublas/vector.hpp>
#include <boost/numeric/cublas/blas.hpp>
#include "handle.hpp"
#include "error.hpp"
#include <cublas_v2.h>

namespace boost {
namespace numeric {
namespace cublas {

template <typename T>
vector<T>::vector()
:
	_size(),
	_elements()
{
}

template <typename T>
vector<T>::vector(const std::size_t size)
:
	_size(size),
	_elements(cuda::make_container<T>(_size))
{
}

template <typename T>
vector<T>::vector(const ublas::vector<T>& vector)
:
	_size(vector.size()),
	_elements(cuda::make_container<T>(_size))
{
	const cublasStatus_t status = cublasSetVector(_size, sizeof(T), &vector.data()[0], 1, _elements.get(), 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

template <typename T>
vector<T>::vector(const cublas::vector<T>& vector)
:
	_size(vector.size()),
	_elements(cuda::make_container<T>(_size))
{
	copy(vector, *this);
}

//template <typename T>
//vector<T>::vector(cublas::vector<T>&& vector)
//:
//	_size(),
//	_elements()
//{
//	std::swap(_size, vector._size);
//	std::swap(_elements, vector._elements);
//}

template <typename T>
vector<T>&
vector<T>::operator=(const cublas::vector<T>& vector)
{
	_size = vector._size;
	_elements = cuda::make_container<T>(_size);
	copy(vector, *this);
	return *this;
}

//template <typename T>
//vector<T>&
//vector<T>::operator=(cublas::vector<T>&& vector) noexcept
//{
//	_size = 0;
//	_elements.reset();
//	std::swap(_size, vector._size);
//	std::swap(_elements, vector._elements);
//	return *this;
//}

template <typename T>
vector<T>::operator ublas::vector<T>() const
{
	ublas::vector<T> vector(_size);
	const cublasStatus_t status = cublasGetVector(_size, sizeof(T), _elements.get(), 1, &vector.data()[0], 1);
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
	return std::move(vector);
}

template <typename T>
vector<T>&
vector<T>::operator+=(const vector<T>& vector)
{
	axpy(T(+1), vector, *this);
	return *this;
}

template <typename T>
vector<T>&
vector<T>::operator-=(const vector<T>& vector)
{
	axpy(T(-1), vector, *this);
	return *this;
}

template <typename T>
vector<T>&
vector<T>::operator*=(const T& value)
{
	scal(value, *this);
	return *this;
}

template <typename T>
vector<T>&
vector<T>::operator/=(const T& value)
{
	scal(T(1) / value, *this);
	return *this;
}

template <typename T>
T
operator*(const vector<T>& vector1, const vector<T>& vector2)
{
	return dot(vector1, vector2);
}

template class vector<real32_t>;
template class vector<real64_t>;
template class vector<complex32_t>;
template class vector<complex64_t>;

template real32_t operator*(const vector<real32_t>& vector1, const vector<real32_t>& vector2);
template real64_t operator*(const vector<real64_t>& vector1, const vector<real64_t>& vector2);
template complex32_t operator*(const vector<complex32_t>& vector1, const vector<complex32_t>& vector2);
template complex64_t operator*(const vector<complex64_t>& vector1, const vector<complex64_t>& vector2);

}
}
}
