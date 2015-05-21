/*
 * test_matrix.cpp
 *
 *  Created on: 06.02.2013
 *      Author: mike_gresens
 */

#include "type.hpp"
#include "generator.hpp"
#include "check.hpp"
#include <boost/numeric/cublas/matrix.hpp>
#include <boost/test/unit_test.hpp>

namespace boost {
namespace numeric {
namespace cublas {
namespace test {

BOOST_AUTO_TEST_SUITE(test_matrix)

constexpr std::size_t M = 20;
constexpr std::size_t N = 10;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_plus, T, types_t)
{
	generator<T> generate;

	const ublas::matrix<T> matrix1u = generate(M, N);
	const ublas::matrix<T> matrix2u = generate(M, N);

	cublas::matrix<T> matrix1c, matrix2c;

	BOOST_REQUIRE_NO_THROW(matrix1c = matrix1u);
	BOOST_REQUIRE_NO_THROW(matrix2c = matrix2u);

	ublas::matrix<T> resultu, resultc;

	BOOST_REQUIRE_NO_THROW(resultu = matrix1u + matrix2u);
	BOOST_REQUIRE_NO_THROW(resultc = matrix1c + matrix2c);

	BOOST_CHECK_MATRIX(resultu, resultc);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_minus, T, types_t)
{
	generator<T> generate;

	const ublas::matrix<T> matrix1u = generate(M, N);
	const ublas::matrix<T> matrix2u = generate(M, N);

	cublas::matrix<T> matrix1c, matrix2c;

	BOOST_REQUIRE_NO_THROW(matrix1c = matrix1u);
	BOOST_REQUIRE_NO_THROW(matrix2c = matrix2u);

	ublas::matrix<T> resultu, resultc;

	BOOST_REQUIRE_NO_THROW(resultu = matrix1u - matrix2u);
	BOOST_REQUIRE_NO_THROW(resultc = matrix1c - matrix2c);

	BOOST_CHECK_MATRIX(resultu, resultc);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_multiply, T, types_t)
{
	generator<T> generate;

	const T scalar = generate();
	const ublas::matrix<T> matrixu = generate(M, N);

	cublas::matrix<T> matrixc;

	BOOST_REQUIRE_NO_THROW(matrixc = matrixu);

	ublas::matrix<T> resultu, resultc;

	BOOST_REQUIRE_NO_THROW(resultu = matrixu * scalar);
	BOOST_REQUIRE_NO_THROW(resultc = matrixc * scalar);

	BOOST_CHECK_MATRIX(resultu, resultc);

	BOOST_REQUIRE_NO_THROW(resultu = scalar * matrixu);
	BOOST_REQUIRE_NO_THROW(resultc = scalar * matrixc);

	BOOST_CHECK_MATRIX(resultu, resultc);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_divide, T, types_t)
{
	generator<T> generate;

	const T scalar = generate();
	const ublas::matrix<T> matrixu = generate(M, N);

	cublas::matrix<T> matrixc;

	BOOST_REQUIRE_NO_THROW(matrixc = matrixu);

	ublas::matrix<T> resultu, resultc;

	BOOST_REQUIRE_NO_THROW(resultu = matrixu / scalar);
	BOOST_REQUIRE_NO_THROW(resultc = matrixc / scalar);

	BOOST_CHECK_MATRIX(resultu, resultc);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_product, T, types_t)
{
	generator<T> generate;

	const ublas::matrix<T> matrix1u = generate(M + 5, N);
	const ublas::matrix<T> matrix2u = generate(N, M - 5);

	cublas::matrix<T> matrix1c, matrix2c;

	BOOST_REQUIRE_NO_THROW(matrix1c = matrix1u);
	BOOST_REQUIRE_NO_THROW(matrix2c = matrix2u);

	ublas::matrix<T> resultu, resultc;

	BOOST_REQUIRE_NO_THROW(resultu = ublas::prod(matrix1u, matrix2u));
	BOOST_REQUIRE_NO_THROW(resultc = matrix1c * matrix2c);

	BOOST_CHECK_MATRIX(resultu, resultc);
}

BOOST_AUTO_TEST_SUITE_END()

}
}
}
}
