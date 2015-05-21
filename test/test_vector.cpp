/*
 * test_vector.cpp
 *
 *  Created on: 06.02.2013
 *      Author: mike_gresens
 */

#include <boost/numeric/cublas/vector.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

namespace boost {
namespace numeric {
namespace cublas {
namespace test {

BOOST_AUTO_TEST_SUITE(test_vector)

typedef typename boost::mpl::list<real32_t, real64_t, complex32_t, complex64_t>::type values_t;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_plus, value_t, values_t)
{
	ublas::vector<value_t> vector1u(3);
	vector1u <<= +1, +2, -3;

	ublas::vector<value_t> vector2u(3);
	vector2u <<= +4, -5, +6;

	const cublas::vector<value_t> vector1c = vector1u;
	const cublas::vector<value_t> vector2c = vector2u;

	ublas::vector<value_t> resultu(3);

	BOOST_REQUIRE_NO_THROW(resultu = vector1u + vector2u);

	ublas::vector<value_t> resultc(3);

	BOOST_REQUIRE_NO_THROW(resultc = vector1c + vector2c);

	BOOST_CHECK_EQUAL_COLLECTIONS(resultu.begin(), resultu.end(), resultc.begin(), resultc.end());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_minus, value_t, values_t)
{
	ublas::vector<value_t> vector1u(3);
	vector1u <<= +1, +2, -3;

	ublas::vector<value_t> vector2u(3);
	vector2u <<= +4, -5, +6;

	const cublas::vector<value_t> vector1c = vector1u;
	const cublas::vector<value_t> vector2c = vector2u;

	ublas::vector<value_t> resultu(3);

	BOOST_REQUIRE_NO_THROW(resultu = vector1u - vector2u);

	ublas::vector<value_t> resultc(3);

	BOOST_REQUIRE_NO_THROW(resultc = vector1c - vector2c);

	BOOST_CHECK_EQUAL_COLLECTIONS(resultu.begin(), resultu.end(), resultc.begin(), resultc.end());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_multiply, value_t, values_t)
{
	constexpr value_t scalar = -5;

	ublas::vector<value_t> vector1u(3);
	vector1u <<= +1, +2, -3;

	const cublas::vector<value_t> vector1c = vector1u;

	ublas::vector<value_t> resultu(3);

	BOOST_REQUIRE_NO_THROW(resultu = vector1u * scalar);

	ublas::vector<value_t> resultc(3);

	BOOST_REQUIRE_NO_THROW(resultc = vector1c * scalar);

	BOOST_CHECK_EQUAL_COLLECTIONS(resultu.begin(), resultu.end(), resultc.begin(), resultc.end());

	BOOST_REQUIRE_NO_THROW(resultu = scalar * vector1u);

	BOOST_REQUIRE_NO_THROW(resultc = scalar * vector1c);

	BOOST_CHECK_EQUAL_COLLECTIONS(resultu.begin(), resultu.end(), resultc.begin(), resultc.end());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_divide, value_t, values_t)
{
	constexpr value_t scalar = -5;

	ublas::vector<value_t> vector1u(3);
	vector1u <<= +1, +2, -3;

	const cublas::vector<value_t> vector1c = vector1u;

	ublas::vector<value_t> resultu(3);

	BOOST_REQUIRE_NO_THROW(resultu = vector1u / scalar);

	ublas::vector<value_t> resultc(3);

	BOOST_REQUIRE_NO_THROW(resultc = vector1c / scalar);

	for (std::size_t i = 0; i < 3; ++i)
		BOOST_CHECK_CLOSE(std::abs(resultu[i]), std::abs(resultc[i]), 1e-8);
}
/*
BOOST_AUTO_TEST_CASE_TEMPLATE(test_minus, value_t, values_t)
{
	const gpu::vector<value_t> vector1 {{+1, +2, -3}};
	const gpu::vector<value_t> vector2 {{+4, -5, +6}};
	const cpu::vector<value_t> expected {{-3, +7, -9}};

	cpu::vector<value_t> result(0);
	BOOST_REQUIRE_NO_THROW(result = vector1 - vector2);

	BOOST_CHECK_EQUAL_COLLECTIONS((*expected).begin(), (*expected).end(), (*result).begin(), (*result).end());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_multiply, value_t, values_t)
{
	const gpu::vector<value_t> vector {{+1, +2, -3}};
	const cpu::vector<value_t> expected {{-3, -6, +9}};

	cpu::vector<value_t> result(0);
	BOOST_REQUIRE_NO_THROW(result = vector * value_t(-3));

	BOOST_CHECK_EQUAL_COLLECTIONS((*expected).begin(), (*expected).end(), (*result).begin(), (*result).end());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_divide, value_t, values_t)
{
	const gpu::vector<value_t> vector {{-3, -6, +9}};
	const cpu::vector<value_t> expected {{+1, +2, -3}};

	cpu::vector<value_t> result(0);
	BOOST_REQUIRE_NO_THROW(result = vector / value_t(-3));

	BOOST_CHECK_EQUAL_COLLECTIONS((*expected).begin(), (*expected).end(), (*result).begin(), (*result).end());
}
*/

/*
BOOST_AUTO_TEST_CASE(test_minus_assign)
{
	constexpr vector3_t expected {{-3, +7, -9}};
	vector3_t vector = vector1;
	BOOST_CHECK_EQUAL(vector -= vector2, expected);
}

BOOST_AUTO_TEST_CASE(test_multiplies_assign_vector)
{
	constexpr vector3_t expected {{+4, -10, -18}};
	vector3_t vector = vector1;
	BOOST_CHECK_EQUAL(vector &= vector2, expected);
}

BOOST_AUTO_TEST_CASE(test_divides_assign_vector)
{
	constexpr vector3_t expected {{value_t(+1) / value_t(+4), value_t(+2) / value_t(-5), value_t(-3) / value_t(+6)}};
	vector3_t vector = vector1;
	BOOST_CHECK_EQUAL(vector |= vector2, expected);
}

BOOST_AUTO_TEST_CASE(test_multiplies_assign_scalar)
{
	constexpr vector3_t expected {{-3, -6, +9}};
	vector3_t vector = vector1;
	BOOST_CHECK_EQUAL(vector *= scalar, expected);
}

BOOST_AUTO_TEST_CASE(test_divides_assign_scalar)
{
	constexpr vector3_t expected {{value_t(+1) / value_t(-3), value_t(+2) / value_t(-3), value_t(-3) / value_t(-3)}};
	vector3_t vector = vector1;
	BOOST_CHECK_EQUAL(vector /= scalar, expected);
}

BOOST_AUTO_TEST_CASE(test_foo)
{
	BOOST_CHECK_EQUAL(+vector1, vector1);
}

BOOST_AUTO_TEST_CASE(test_negate)
{
	constexpr vector3_t expected {{-1, -2, +3}};
	BOOST_CHECK_EQUAL(-vector1, expected);
}

BOOST_AUTO_TEST_CASE(test_plus)
{
	constexpr vector3_t expected {{+5, -3, +3}};
	BOOST_CHECK_EQUAL(vector1 + vector2, expected);
}

BOOST_AUTO_TEST_CASE(test_minus)
{
	constexpr vector3_t expected {{-3, +7, -9}};
	BOOST_CHECK_EQUAL(vector1 - vector2, expected);
}

BOOST_AUTO_TEST_CASE(test_dot)
{
	constexpr value_t expected = -24;
	BOOST_CHECK_EQUAL(vector1 * vector2, expected);
}

BOOST_AUTO_TEST_CASE(test_multiplies_vector)
{
	constexpr vector3_t expected {{+4, -10, -18}};
	BOOST_CHECK_EQUAL(vector1 & vector2, expected);
}

BOOST_AUTO_TEST_CASE(test_divides_vector)
{
	constexpr vector3_t expected {{value_t(+1) / value_t(+4), value_t(+2) / value_t(-5), value_t(-3) / value_t(+6)}};
	BOOST_CHECK_EQUAL(vector1 | vector2, expected);
}

BOOST_AUTO_TEST_CASE(test_multiplies_scalar)
{
	constexpr vector3_t expected {{-3, -6, +9}};
	BOOST_CHECK_EQUAL(vector1 * scalar, expected);
	BOOST_CHECK_EQUAL(scalar * vector1, expected);
}

BOOST_AUTO_TEST_CASE(test_divides_scalar)
{
	constexpr vector3_t expected {{value_t(+1) / value_t(-3), value_t(+2) / value_t(-3), value_t(-3) / value_t(-3)}};
	BOOST_CHECK_EQUAL(vector1 / scalar, expected);
}

BOOST_AUTO_TEST_CASE(test_cross)
{
	constexpr vector3_t expected {{-3, -18, -13}};
	BOOST_CHECK_EQUAL(vector1 % vector2, expected);
}

BOOST_AUTO_TEST_CASE(test_length)
{
	static const value_t expected = std::sqrt(value_t(14));
	BOOST_CHECK_EQUAL(length(vector1), expected);
}

BOOST_AUTO_TEST_CASE(test_normalize)
{
	static const value_t length = std::sqrt(value_t(14));
	static const vector3_t expected {{+1 / length, +2 / length, -3 / length}};
	BOOST_CHECK_EQUAL(normalize(vector1), expected);
}

BOOST_AUTO_TEST_CASE(test_size)
{
	constexpr std::size_t expected = 3;
	BOOST_CHECK_EQUAL(vector1.size(), expected);
}

BOOST_AUTO_TEST_CASE(test_equal_to)
{
	BOOST_CHECK_EQUAL(vector1 == vector1, true);
	BOOST_CHECK_EQUAL(vector1 == vector2, false);
}

BOOST_AUTO_TEST_CASE(test_not_equal_to)
{
	BOOST_CHECK_EQUAL(vector1 != vector1, false);
	BOOST_CHECK_EQUAL(vector1 != vector2, true);
}
*/
BOOST_AUTO_TEST_SUITE_END()

}
}
}
}
