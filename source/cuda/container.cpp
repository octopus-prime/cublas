/*
 * container.cpp
 *
 *  Created on: 14.05.2015
 *      Author: mike_gresens
 */

//#include <boost/numeric/type.hpp>
#include <boost/numeric/cuda/container.hpp>
#include "error.hpp"
#include <cuda_runtime.h>

namespace boost {
namespace numeric {
namespace cuda {

template <typename T>
void
deleter<T>::operator()(T* elements) const
{
	cudaFree(elements);
}

template <typename T>
container<T>
make_container(const std::size_t size)
{
	T* elements;
	const cudaError error = cudaMalloc(&elements, size * sizeof(T));
	if (error != cudaSuccess)
		throw std::system_error(error, category, __PRETTY_FUNCTION__);
	return container<T>(elements);
}

template struct deleter<int>;
template struct deleter<real32_t>;
template struct deleter<real64_t>;
template struct deleter<complex32_t>;
template struct deleter<complex64_t>;

template container<int> make_container(const std::size_t size);
template container<real32_t> make_container(const std::size_t size);
template container<real64_t> make_container(const std::size_t size);
template container<complex32_t> make_container(const std::size_t size);
template container<complex64_t> make_container(const std::size_t size);

}
}
}
