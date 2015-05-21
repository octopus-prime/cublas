/*
 * type.hpp
 *
 *  Created on: 21.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <boost/numeric/type.hpp>
#include <boost/mpl/list.hpp>

namespace boost {
namespace numeric {

typedef typename boost::mpl::list<real32_t, real64_t, complex32_t, complex64_t>::type types_t;

}
}
