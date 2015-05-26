/*
 * type_trait.hpp
 *
 *  Created on: 22.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <cuComplex.h>

namespace boost {
namespace numeric {

template <typename T>
struct type_trait
{
	typedef T type;
};

template <>
struct type_trait<complex32_t>
{
	typedef cuFloatComplex type;
};

template <>
struct type_trait<complex64_t>
{
	typedef cuDoubleComplex type;
};

}
}
