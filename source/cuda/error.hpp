/*
 * error.hpp
 *
 *  Created on: 10.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <system_error>

namespace boost {
namespace numeric {
namespace cuda {

class category_impl
:
	public std::error_category
{
public:
	category_impl(int) {}

    virtual const char*
    name() const noexcept override;

    virtual std::string
    message(int) const override;
};

extern const category_impl category;

}
}
}
