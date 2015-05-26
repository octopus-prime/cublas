/*
 * example.cpp
 *
 *  Created on: 14.05.2015
 *      Author: mike_gresens
 */

#include "generator.hpp"
#include "runner.hpp"
#include <boost/numeric/cublas/blas.hpp>
#include <boost/numeric/cusolver/lu.hpp>
#include <boost/numeric/cusolver/qr.hpp>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <iostream>

using namespace boost::numeric;

/*       | 1 2 3 |
 *   A = | 4 5 6 |
 *       | 2 1 1 |
 *
 *   x = (1 1 1)'
 *   b = (6 15 4)'
 */
static void
test_solve()
{
	using cusolver::lu::solver;
	typedef real64_t value_t;
	constexpr std::size_t N = 3;

	ublas::matrix<value_t> A(N, N);
//	A <<= 2, 0, 2, 1, 2, 2, 0, 2, 3;
	A <<= 1, 2, 3, 4, 5, 6, 2, 1, 1;

	const solver<value_t> solve(A);

	ublas::vector<value_t> y(N);
//	y <<= 8, 9, 7;
	y <<= 6, 15, 4;

	const ublas::vector<value_t> x = solve(y);

	std::cout << "x = " << x << std::endl;
}

static void
test_invert()
{
	using cusolver::lu::inverter;
	typedef real64_t value_t;
	constexpr std::size_t N = 4;

	ublas::matrix<value_t> A(N, N);
	A <<= 1, 3, -1, 4, 2, 5, -1, 3, 0, 4, -3, 1, -3, 1, -5, -2;

	const inverter<value_t> invert(A);

	const ublas::matrix<value_t> I = invert();

	std::cout << "I = " << I << std::endl;
}

// Given:
// a, b, A1, A2, A3, y1, y2

// Wanted:
// r

// Calc:
// A = a * A1 * A2 + b * A3
// x1 = y1 / A
// X2 = y2 / A
// r = x1 * x2

static void
test()
{
	typedef complex32_t value_t;

	constexpr std::size_t N = 1000;

	generator<value_t> generate;

	const value_t a = generate();
	const value_t b = generate();

	const ublas::matrix<value_t> A1 = generate(N, N);
	const ublas::matrix<value_t> A2 = generate(N, N);
	const ublas::matrix<value_t> A3 = generate(N, N);

	const ublas::vector<value_t> y1 = generate(N);
	const ublas::vector<value_t> y2 = generate(N);

	const auto cpu = [&]() -> value_t
	{
		ublas::matrix<value_t> A(A3);
		ublas::blas_3::gmm(A, b, a, A1, A2);

		ublas::permutation_matrix<std::size_t> P(N);
		ublas::lu_factorize(A, P);

		ublas::vector<value_t> x1(y1);
		ublas::lu_substitute(A, P, x1);

		ublas::vector<value_t> x2(y2);
		ublas::lu_substitute(A, P, x2);

		return ublas::blas_1::dot(x1, x2);
	};

	const auto gpu = [&]() -> value_t
	{
		cublas::matrix<value_t> A(A3);
		cublas::gemm<value_t>(a, A1, A2, b, A);

		const cusolver::lu::solver<value_t> solve(std::move(A));
		const cublas::vector<value_t> x1 = solve(y1);
		const cublas::vector<value_t> x2 = solve(y2);

		return cublas::dot(x1, x2);
	};

	runner("gpu")(gpu);
	runner("cpu")(cpu);
}

int
main()
{
//	try
//	{
		test_solve();
		test_invert();
		test();
//	}
//	catch (const std::exception& e)
//	{
//		std::cerr << "error: " << e.what() << std::endl;
//	}
	return 0;
}
