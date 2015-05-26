/*
 * test_blas.cpp
 *
 *  Created on: 26.05.2015
 *      Author: mike_gresens
 */

#include "type.hpp"
#include "generator.hpp"
#include "check.hpp"
#include <boost/numeric/cublas/blas.hpp>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/range/algorithm/min_element.hpp>
#include <boost/test/unit_test.hpp>

namespace boost {
namespace numeric {
namespace cublas {

BOOST_AUTO_TEST_SUITE(test_blas)

constexpr std::size_t N = 100;

BOOST_AUTO_TEST_CASE_TEMPLATE(test_amax, T, types_t)
{
	generator<T> generate;

	const ublas::vector<T> vectoru = generate(N);

	cublas::vector<T> vectorc;

	BOOST_REQUIRE_NO_THROW(vectorc = vectoru);

	T resultu, resultc;

	BOOST_REQUIRE_NO_THROW(resultu = ublas::blas_1::amax(vectoru));
	BOOST_REQUIRE_NO_THROW(resultc = std::abs(vectoru[amax(vectorc)]));

	BOOST_CHECK_EQUAL(resultu, resultc);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(test_amin, T, types_t)
{
	generator<T> generate;

	const ublas::vector<T> vectoru = generate(N);

	cublas::vector<T> vectorc;

	BOOST_REQUIRE_NO_THROW(vectorc = vectoru);

	T resultu, resultc;

	BOOST_REQUIRE_NO_THROW(resultu = *boost::min_element(vectoru, [](const T& v1, const T& v2){return std::abs(v1) < std::abs(v2);}));
	BOOST_REQUIRE_NO_THROW(resultc = vectoru[amin(vectorc)]);

	BOOST_CHECK_EQUAL(resultu, resultc);
}

BOOST_AUTO_TEST_SUITE_END()

}
}
}
