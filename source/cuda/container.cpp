/*
 * container.cpp
 *
 *  Created on: 14.05.2015
 *      Author: mike_gresens
 */

#include <boost/numeric/type.hpp>
#include <boost/numeric/cuda/container.hpp>
#include "error.hpp"
#include <cuda_runtime.h>

namespace boost {
namespace numeric {
namespace cuda {
namespace detail {

template <typename T, typename P = typename container<T>::pointer>
P
alloc(const std::size_t size)
{
	T* elements;
	const cudaError error = cudaMalloc(&elements, size * sizeof(T));
	if (error != cudaSuccess)
		throw std::system_error(error, category, __func__);
	return P(elements);
}

template <typename T, typename P = typename container<T>::pointer>
void
memcopy(P& dest, const P& source, const std::size_t size)
{
	const cudaError error = cudaMemcpy(dest.get(), source.get(), size * sizeof(T),  cudaMemcpyDeviceToDevice);
	if (error != cudaSuccess)
		throw std::system_error(error, category, __func__);
}

}

template <typename T>
container<T>::container()
:
	_size(),
	_elements()
{
}

template <typename T>
container<T>::container(const std::size_t size)
:
	_size(size),
	_elements(detail::alloc<T>(_size))
{
}

template <typename T>
container<T>::container(const container<T>& container)
:
	_size(container.size()),
	_elements(detail::alloc<T>(_size))
{
	detail::memcopy<T>(_elements, container._elements, _size);
}

template <typename T>
container<T>::container(container<T>&& container)
:
	_size(),
	_elements()
{
	std::swap(_size, container._size);
	std::swap(_elements, container._elements);
}

template <typename T>
container<T>&
container<T>::operator=(const container<T>& container)
{
	_size = container.size();
	_elements = detail::alloc<T>(_size);
	detail::memcopy<T>(_elements, container._elements, _size);
	return *this;
}

template <typename T>
container<T>&
container<T>::operator=(container<T>&& container) noexcept
{
	_size = 0;
	_elements.release();
	std::swap(_size, container._size);
	std::swap(_elements, container._elements);
	return *this;
}

template <typename T>
void
container<T>::deleter::operator()(T* elements) const
{
	cudaFree(elements);
}

template class container<int>;
template class container<real32_t>;
template class container<real64_t>;
template class container<complex32_t>;
template class container<complex64_t>;

}
}
}
