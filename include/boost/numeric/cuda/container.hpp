/*
 * container.hpp
 *
 *  Created on: 13.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <boost/numeric/type.hpp>
#include <memory>

namespace boost {
namespace numeric {
namespace cuda {

template <typename T>
class container
{
public:
	struct deleter
	{
		void operator()(T* elements) const;
	};

	typedef std::unique_ptr<T[], deleter> pointer;

	container();
	container(const std::size_t size);
	container(const container<T>& container);
	container(container<T>&& container);

	container<T>&
	operator=(const container<T>& container);

	container<T>&
	operator=(container<T>&& container) noexcept;

	std::size_t
	size() const noexcept;

	const pointer&
	operator*() const noexcept;

private:
	std::size_t _size;
	pointer _elements;
};

template <typename T>
std::size_t
container<T>::size() const noexcept
{
	return _size;
}

template <typename T>
const typename container<T>::pointer&
container<T>::operator*() const noexcept
{
	return _elements;
}

}
}
}
