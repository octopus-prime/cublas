/*
 * test_vector.cpp
 *
 *  Created on: 06.02.2013
 *      Author: mike_gresens
 */

#include "type.hpp"
#include "generator.hpp"
#include "check.hpp"
#include <boost/numeric/cublas/vector.hpp>
#include <boost/test/unit_test.hpp>

namespace boost {
namespace numeric {
namespace cublas {
namespace test {

BOOST_AUTO_TEST_SUITE(test_vector)

constexpr std::size_t N = 100;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_plus, T, types_t)
{
	generator<T> generate;

	const ublas::vector<T> vector1u = generate(N);
	const ublas::vector<T> vector2u = generate(N);

	cublas::vector<T> vector1c, vector2c;

	BOOST_REQUIRE_NO_THROW(vector1c = vector1u);
	BOOST_REQUIRE_NO_THROW(vector2c = vector2u);

	ublas::vector<T> resultu, resultc;

	BOOST_REQUIRE_NO_THROW(resultu = vector1u + vector2u);
	BOOST_REQUIRE_NO_THROW(resultc = vector1c + vector2c);

	BOOST_CHECK_VECTOR(resultu, resultc);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_minus, T, types_t)
{
	generator<T> generate;

	const ublas::vector<T> vector1u = generate(N);
	const ublas::vector<T> vector2u = generate(N);

	cublas::vector<T> vector1c, vector2c;

	BOOST_REQUIRE_NO_THROW(vector1c = vector1u);
	BOOST_REQUIRE_NO_THROW(vector2c = vector2u);

	ublas::vector<T> resultu, resultc;

	BOOST_REQUIRE_NO_THROW(resultu = vector1u - vector2u);
	BOOST_REQUIRE_NO_THROW(resultc = vector1c - vector2c);

	BOOST_CHECK_VECTOR(resultu, resultc);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_multiply, T, types_t)
{
	generator<T> generate;

	const T scalar = generate();
	const ublas::vector<T> vectoru = generate(N);

	cublas::vector<T> vectorc;

	BOOST_REQUIRE_NO_THROW(vectorc = vectoru);

	ublas::vector<T> resultu, resultc;

	BOOST_REQUIRE_NO_THROW(resultu = vectoru * scalar);
	BOOST_REQUIRE_NO_THROW(resultc = vectorc * scalar);

	BOOST_CHECK_VECTOR(resultu, resultc);

	BOOST_REQUIRE_NO_THROW(resultu = scalar * vectoru);
	BOOST_REQUIRE_NO_THROW(resultc = scalar * vectorc);

	BOOST_CHECK_VECTOR(resultu, resultc);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_divide, T, types_t)
{
	generator<T> generate;

	const T scalar = generate();
	const ublas::vector<T> vectoru = generate(N);

	cublas::vector<T> vectorc;

	BOOST_REQUIRE_NO_THROW(vectorc = vectoru);

	ublas::vector<T> resultu, resultc;

	BOOST_REQUIRE_NO_THROW(resultu = vectoru / scalar);
	BOOST_REQUIRE_NO_THROW(resultc = vectorc / scalar);

	BOOST_CHECK_VECTOR(resultu, resultc);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_product, T, types_t)
{
	generator<T> generate;

	const ublas::vector<T> vector1u = generate(N);
	const ublas::vector<T> vector2u = generate(N);

	cublas::vector<T> vector1c, vector2c;

	BOOST_REQUIRE_NO_THROW(vector1c = vector1u);
	BOOST_REQUIRE_NO_THROW(vector2c = vector2u);

	T resultu, resultc;

	BOOST_REQUIRE_NO_THROW(resultu = ublas::inner_prod(vector1u, vector2u));
	BOOST_REQUIRE_NO_THROW(resultc = vector1c * vector2c);

	BOOST_CHECK_CLOSE(std::abs(resultu), std::abs(resultc), tolerance);
}

BOOST_AUTO_TEST_SUITE_END()

}
}
}
}
