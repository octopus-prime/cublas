/*
 * check.hpp
 *
 *  Created on: 21.05.2015
 *      Author: mike_gresens
 */

#pragma once

#define BOOST_CHECK_VECTOR(U, C) \
	BOOST_REQUIRE_EQUAL(U.size(), C.size()); \
	for (std::size_t i = 0; i < U.size(); ++i) \
		BOOST_CHECK_CLOSE(std::abs(U[i]), std::abs(C[i]), 1e-4);

#define BOOST_CHECK_MATRIX(U, C) \
	BOOST_REQUIRE_EQUAL(U.size1(), C.size1()); \
	BOOST_REQUIRE_EQUAL(U.size2(), C.size2()); \
	for (std::size_t i = 0; i < U.size1(); ++i) \
		for (std::size_t j = 0; j < U.size2(); ++j) \
			BOOST_CHECK_CLOSE(std::abs(U(i,j)), std::abs(C(i,j)), 1e-4);
