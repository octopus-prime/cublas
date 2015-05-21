/*
 * test_matrix.cpp
 *
 *  Created on: 06.02.2013
 *      Author: mike_gresens
 */

#include <boost/numeric/cublas/matrix.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/test/unit_test.hpp>
#include <boost/mpl/list.hpp>

namespace boost {
namespace numeric {
namespace cublas {
namespace test {

BOOST_AUTO_TEST_SUITE(test_matrix)

typedef typename boost::mpl::list<real32_t, real64_t, complex32_t, complex64_t>::type values_t;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_plus, value_t, values_t)
{
	ublas::matrix<value_t> matrix1u(3, 3);
	matrix1u <<= +1, +2, -3, -4, +5, -6, -7, +8, +9;

	ublas::matrix<value_t> matrix2u(3, 3);
	matrix2u <<= +4, -5, +6, -7, +8, -9, +1, -2, +3;

	const cublas::matrix<value_t> matrix1c = matrix1u;
	const cublas::matrix<value_t> matrix2c = matrix2u;

	ublas::matrix<value_t> resultu;

	BOOST_REQUIRE_NO_THROW(resultu = matrix1u + matrix2u);

	ublas::matrix<value_t> resultc;

	BOOST_REQUIRE_NO_THROW(resultc = matrix1c + matrix2c);

	for (std::size_t i = 0; i < 3; ++i)
		for (std::size_t j = 0; j < 3; ++j)
			BOOST_CHECK_CLOSE(std::abs(resultu(i, j)), std::abs(resultc(i, j)), 1e-5);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_minus, value_t, values_t)
{
	ublas::matrix<value_t> matrix1u(3, 3);
	matrix1u <<= +1, +2, -3, -4, +5, -6, -7, +8, +9;

	ublas::matrix<value_t> matrix2u(3, 3);
	matrix2u <<= +4, -5, +6, -7, +8, -9, +1, -2, +3;

	const cublas::matrix<value_t> matrix1c = matrix1u;
	const cublas::matrix<value_t> matrix2c = matrix2u;

	ublas::matrix<value_t> resultu;

	BOOST_REQUIRE_NO_THROW(resultu = matrix1u - matrix2u);

	ublas::matrix<value_t> resultc;

	BOOST_REQUIRE_NO_THROW(resultc = matrix1c - matrix2c);

	for (std::size_t i = 0; i < 3; ++i)
		for (std::size_t j = 0; j < 3; ++j)
			BOOST_CHECK_CLOSE(std::abs(resultu(i, j)), std::abs(resultc(i, j)), 1e-5);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_multiply, value_t, values_t)
{
	constexpr value_t scalar = -5;

	ublas::matrix<value_t> matrix1u(3, 3);
	matrix1u <<= +1, +2, -3, -4, +5, -6, -7, +8, +9;

	const cublas::matrix<value_t> matrix1c = matrix1u;

	ublas::matrix<value_t> resultu;

	BOOST_REQUIRE_NO_THROW(resultu = matrix1u * scalar);

	ublas::matrix<value_t> resultc;

	BOOST_REQUIRE_NO_THROW(resultc = matrix1c * scalar);

	for (std::size_t i = 0; i < 3; ++i)
		for (std::size_t j = 0; j < 3; ++j)
			BOOST_CHECK_CLOSE(std::abs(resultu(i, j)), std::abs(resultc(i, j)), 1e-5);

	BOOST_REQUIRE_NO_THROW(resultu = scalar * matrix1u);

	BOOST_REQUIRE_NO_THROW(resultc = scalar * matrix1c);

	for (std::size_t i = 0; i < 3; ++i)
		for (std::size_t j = 0; j < 3; ++j)
			BOOST_CHECK_CLOSE(std::abs(resultu(i, j)), std::abs(resultc(i, j)), 1e-5);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_divide, value_t, values_t)
{
	constexpr value_t scalar = -5;

	ublas::matrix<value_t> matrix1u(3, 3);
	matrix1u <<= +1, +2, -3, -4, +5, -6, -7, +8, +9;

	const cublas::matrix<value_t> matrix1c = matrix1u;

	ublas::matrix<value_t> resultu;

	BOOST_REQUIRE_NO_THROW(resultu = matrix1u / scalar);

	ublas::matrix<value_t> resultc;

	BOOST_REQUIRE_NO_THROW(resultc = matrix1c / scalar);

	for (std::size_t i = 0; i < 3; ++i)
		for (std::size_t j = 0; j < 3; ++j)
			BOOST_CHECK_CLOSE(std::abs(resultu(i, j)), std::abs(resultc(i, j)), 1e-5);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_product, value_t, values_t)
{
	ublas::matrix<value_t> matrix1u(4, 3);
	matrix1u <<= +1, +2, -3, -4, +5, -6, -7, +8, +9, -2, +5, -8;

	ublas::matrix<value_t> matrix2u(3, 2);
	matrix2u <<= +4, -5, +6, -7, +8, -9;

	const cublas::matrix<value_t> matrix1c = matrix1u;
	const cublas::matrix<value_t> matrix2c = matrix2u;

	ublas::matrix<value_t> resultu;

	BOOST_REQUIRE_NO_THROW(resultu = ublas::prod(matrix1u, matrix2u));

	ublas::matrix<value_t> resultc;

	BOOST_REQUIRE_NO_THROW(resultc = matrix1c * matrix2c);

	for (std::size_t i = 0; i < 4; ++i)
		for (std::size_t j = 0; j < 2; ++j)
			BOOST_CHECK_CLOSE(std::abs(resultu(i, j)), std::abs(resultc(i, j)), 1e-5);
}

/*
BOOST_AUTO_TEST_CASE_TEMPLATE(test_minus, value_t, values_t)
{
	const gpu::matrix<value_t> matrix1 {{+1, +2, -3}};
	const gpu::matrix<value_t> matrix2 {{+4, -5, +6}};
	const cpu::matrix<value_t> expected {{-3, +7, -9}};

	cpu::matrix<value_t> result(0);
	BOOST_REQUIRE_NO_THROW(result = matrix1 - matrix2);

	BOOST_CHECK_EQUAL_COLLECTIONS((*expected).begin(), (*expected).end(), (*result).begin(), (*result).end());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_multiply, value_t, values_t)
{
	const gpu::matrix<value_t> matrix {{+1, +2, -3}};
	const cpu::matrix<value_t> expected {{-3, -6, +9}};

	cpu::matrix<value_t> result(0);
	BOOST_REQUIRE_NO_THROW(result = matrix * value_t(-3));

	BOOST_CHECK_EQUAL_COLLECTIONS((*expected).begin(), (*expected).end(), (*result).begin(), (*result).end());
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_divide, value_t, values_t)
{
	const gpu::matrix<value_t> matrix {{-3, -6, +9}};
	const cpu::matrix<value_t> expected {{+1, +2, -3}};

	cpu::matrix<value_t> result(0);
	BOOST_REQUIRE_NO_THROW(result = matrix / value_t(-3));

	BOOST_CHECK_EQUAL_COLLECTIONS((*expected).begin(), (*expected).end(), (*result).begin(), (*result).end());
}
*/

/*
BOOST_AUTO_TEST_CASE(test_minus_assign)
{
	constexpr matrix3_t expected {{-3, +7, -9}};
	matrix3_t matrix = matrix1;
	BOOST_CHECK_EQUAL(matrix -= matrix2, expected);
}

BOOST_AUTO_TEST_CASE(test_multiplies_assign_matrix)
{
	constexpr matrix3_t expected {{+4, -10, -18}};
	matrix3_t matrix = matrix1;
	BOOST_CHECK_EQUAL(matrix &= matrix2, expected);
}

BOOST_AUTO_TEST_CASE(test_divides_assign_matrix)
{
	constexpr matrix3_t expected {{value_t(+1) / value_t(+4), value_t(+2) / value_t(-5), value_t(-3) / value_t(+6)}};
	matrix3_t matrix = matrix1;
	BOOST_CHECK_EQUAL(matrix |= matrix2, expected);
}

BOOST_AUTO_TEST_CASE(test_multiplies_assign_scalar)
{
	constexpr matrix3_t expected {{-3, -6, +9}};
	matrix3_t matrix = matrix1;
	BOOST_CHECK_EQUAL(matrix *= scalar, expected);
}

BOOST_AUTO_TEST_CASE(test_divides_assign_scalar)
{
	constexpr matrix3_t expected {{value_t(+1) / value_t(-3), value_t(+2) / value_t(-3), value_t(-3) / value_t(-3)}};
	matrix3_t matrix = matrix1;
	BOOST_CHECK_EQUAL(matrix /= scalar, expected);
}

BOOST_AUTO_TEST_CASE(test_foo)
{
	BOOST_CHECK_EQUAL(+matrix1, matrix1);
}

BOOST_AUTO_TEST_CASE(test_negate)
{
	constexpr matrix3_t expected {{-1, -2, +3}};
	BOOST_CHECK_EQUAL(-matrix1, expected);
}

BOOST_AUTO_TEST_CASE(test_plus)
{
	constexpr matrix3_t expected {{+5, -3, +3}};
	BOOST_CHECK_EQUAL(matrix1 + matrix2, expected);
}

BOOST_AUTO_TEST_CASE(test_minus)
{
	constexpr matrix3_t expected {{-3, +7, -9}};
	BOOST_CHECK_EQUAL(matrix1 - matrix2, expected);
}

BOOST_AUTO_TEST_CASE(test_dot)
{
	constexpr value_t expected = -24;
	BOOST_CHECK_EQUAL(matrix1 * matrix2, expected);
}

BOOST_AUTO_TEST_CASE(test_multiplies_matrix)
{
	constexpr matrix3_t expected {{+4, -10, -18}};
	BOOST_CHECK_EQUAL(matrix1 & matrix2, expected);
}

BOOST_AUTO_TEST_CASE(test_divides_matrix)
{
	constexpr matrix3_t expected {{value_t(+1) / value_t(+4), value_t(+2) / value_t(-5), value_t(-3) / value_t(+6)}};
	BOOST_CHECK_EQUAL(matrix1 | matrix2, expected);
}

BOOST_AUTO_TEST_CASE(test_multiplies_scalar)
{
	constexpr matrix3_t expected {{-3, -6, +9}};
	BOOST_CHECK_EQUAL(matrix1 * scalar, expected);
	BOOST_CHECK_EQUAL(scalar * matrix1, expected);
}

BOOST_AUTO_TEST_CASE(test_divides_scalar)
{
	constexpr matrix3_t expected {{value_t(+1) / value_t(-3), value_t(+2) / value_t(-3), value_t(-3) / value_t(-3)}};
	BOOST_CHECK_EQUAL(matrix1 / scalar, expected);
}

BOOST_AUTO_TEST_CASE(test_cross)
{
	constexpr matrix3_t expected {{-3, -18, -13}};
	BOOST_CHECK_EQUAL(matrix1 % matrix2, expected);
}

BOOST_AUTO_TEST_CASE(test_length)
{
	static const value_t expected = std::sqrt(value_t(14));
	BOOST_CHECK_EQUAL(length(matrix1), expected);
}

BOOST_AUTO_TEST_CASE(test_normalize)
{
	static const value_t length = std::sqrt(value_t(14));
	static const matrix3_t expected {{+1 / length, +2 / length, -3 / length}};
	BOOST_CHECK_EQUAL(normalize(matrix1), expected);
}

BOOST_AUTO_TEST_CASE(test_size)
{
	constexpr std::size_t expected = 3;
	BOOST_CHECK_EQUAL(matrix1.size(), expected);
}

BOOST_AUTO_TEST_CASE(test_equal_to)
{
	BOOST_CHECK_EQUAL(matrix1 == matrix1, true);
	BOOST_CHECK_EQUAL(matrix1 == matrix2, false);
}

BOOST_AUTO_TEST_CASE(test_not_equal_to)
{
	BOOST_CHECK_EQUAL(matrix1 != matrix1, false);
	BOOST_CHECK_EQUAL(matrix1 != matrix2, true);
}
*/
BOOST_AUTO_TEST_SUITE_END()

}
}
}
}
