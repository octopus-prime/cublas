/*
 * test_container.cpp
 *
 *  Created on: 21.05.2015
 *      Author: mike_gresens
 */

#include "type.hpp"
#include <boost/numeric/cuda/container.hpp>
#include <boost/test/unit_test.hpp>

namespace boost {
namespace numeric {
namespace cuda {
namespace test {

BOOST_AUTO_TEST_SUITE(test_container)

constexpr std::size_t N = 100;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_copy_constructor, T, types_t)
{
	container<T> container1(N);
	container<T> container2(container1);

	BOOST_CHECK_EQUAL(N, container1.size());
	BOOST_CHECK_EQUAL(N, container2.size());

	BOOST_CHECK(*container1);
	BOOST_CHECK(*container2);
	BOOST_CHECK(*container1 != *container2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_move_constructor, T, types_t)
{
	container<T> container1(N);
	const container<T> container2(std::move(container1));

	BOOST_CHECK_EQUAL(0, container1.size());
	BOOST_CHECK_EQUAL(N, container2.size());

	BOOST_CHECK(!*container1);
	BOOST_CHECK(*container2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_copy_assignment, T, types_t)
{
	container<T> container1(N);
	container<T> container2;

	container2 = container1;

	BOOST_CHECK_EQUAL(N, container1.size());
	BOOST_CHECK_EQUAL(N, container2.size());

	BOOST_CHECK(*container1);
	BOOST_CHECK(*container2);
	BOOST_CHECK(*container1 != *container2);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_move_assignment, T, types_t)
{
	container<T> container1(N);
	container<T> container2;

	container2 = std::move(container1);

	BOOST_CHECK_EQUAL(0, container1.size());
	BOOST_CHECK_EQUAL(N, container2.size());

	BOOST_CHECK(!*container1);
	BOOST_CHECK(*container2);
}

BOOST_AUTO_TEST_SUITE_END()

}
}
}
}
