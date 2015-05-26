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
struct deleter
{
	void operator()(T* p) const;
};

template <typename T>
using container = std::unique_ptr<T, deleter<T>>;

template <typename T>
container<T>
make_container(const std::size_t size);

}
}
}
