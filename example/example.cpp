/*
 * example.cpp
 *
 *  Created on: 14.05.2015
 *      Author: mike_gresens
 */

#include <boost/numeric/cublas/blas.hpp>
#include <boost/numeric/cusolver/lu.hpp>
#include <boost/numeric/cusolver/qr.hpp>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/range/algorithm/generate.hpp>
#include <chrono>
#include <random>
#include <iostream>

using namespace boost::numeric;

typedef std::chrono::system_clock clk_t;

void test_gemv()
{
	constexpr std::size_t N = 8000;

	const ublas::matrix<real32_t> mu(N, N);
	const ublas::vector<real32_t> vu1(N);

	const cublas::matrix<real32_t> mc(mu);
	const cublas::vector<real32_t> vc1(vu1);
	cublas::vector<real32_t> vc2(N);

	const auto tc0 = clk_t::now();
	cublas::gemv(1.f, mc, vc1, 0.f, vc2);
	const auto tc1 = clk_t::now();

	ublas::vector<real32_t> vu2(N);//vc2);

	const auto tu0 = clk_t::now();
	ublas::blas_2::gmv(vu2, 1.f, 0.f, mu, vu1);
	const auto tu1 = clk_t::now();

	std::cout << (tc1 - tc0).count() << std::endl;
	std::cout << (tu1 - tu0).count() << std::endl;
	std::cout << (tu1 - tu0).count() / (tc1 - tc0).count() << std::endl;
//		std::cout << vu2 << std::endl;
}

void test_gemm()
{
	constexpr std::size_t N = 3000;

	const ublas::matrix<real32_t> mu1(N, N);
	const ublas::matrix<real32_t> mu2(N, N);

	const cublas::matrix<real32_t> mc1(mu1);
	const cublas::matrix<real32_t> mc2(mu2);
	cublas::matrix<real32_t> mc3(N, N);

	const auto tc0 = clk_t::now();
	cublas::gemm(1.f, mc1, mc2, 0.f, mc3);
	const auto tc1 = clk_t::now();

	ublas::matrix<real32_t> mu3(N, N);//vc2);

	const auto tu0 = clk_t::now();
	ublas::blas_3::gmm(mu3, 1.f, 0.f, mu1, mu2);
	const auto tu1 = clk_t::now();

	std::cout << (tc1 - tc0).count() << std::endl;
	std::cout << (tu1 - tu0).count() << std::endl;
	std::cout << (tu1 - tu0).count() / (tc1 - tc0).count() << std::endl;
//		std::cout << vu2 << std::endl;
}

void test_gemm_r()
{
	constexpr std::size_t N = 3000;

	const ublas::matrix<real32_t> mu1(N, N);
	const ublas::matrix<real32_t> mu2(N, N);

	const cublas::matrix<real32_t> mc1(mu1);
	const cublas::matrix<real32_t> mc2(mu2);
	cublas::matrix<real32_t> mc3(N, N);

	const auto tc0 = clk_t::now();
	cublas::gemm(1.f, mc1, mc2, 0.f, mc3);
	const auto tc1 = clk_t::now();

	ublas::matrix<real32_t> mu3(N, N);//vc2);

	const auto tu0 = clk_t::now();
//	ublas::blas_3::gmm(mu3, 1.f, 0.f, mu1, mu2);
	const auto tu1 = clk_t::now();

	std::cout << (tc1 - tc0).count() << std::endl;
	std::cout << (tu1 - tu0).count() << std::endl;
	std::cout << (tu1 - tu0).count() / (tc1 - tc0).count() << std::endl;
//		std::cout << vu2 << std::endl;
}

void test_gemm_c()
{
	constexpr std::size_t N = 3000;

	const ublas::matrix<real32_t> mu1(N, N);
	const ublas::matrix<real32_t> mu2(N, N);

	const cublas::matrix<real32_t> mc1(mu1);
	const cublas::matrix<real32_t> mc2(mu2);
	cublas::matrix<real32_t> mc3(N, N);

	const auto tc0 = clk_t::now();
	cublas::gemm(1.f, mc1, mc2, 0.f, mc3);
	const auto tc1 = clk_t::now();

	ublas::matrix<real32_t> mu3(N, N);//vc2);

	const auto tu0 = clk_t::now();
//	ublas::blas_3::gmm(mu3, 1.f, 0.f, mu1, mu2);
	const auto tu1 = clk_t::now();

	std::cout << (tc1 - tc0).count() << std::endl;
	std::cout << (tu1 - tu0).count() << std::endl;
	std::cout << (tu1 - tu0).count() / (tc1 - tc0).count() << std::endl;
//		std::cout << vu2 << std::endl;
}

/*       | 1 2 3 |
 *   A = | 4 5 6 |
 *       | 2 1 1 |
 *
 *   x = (1 1 1)'
 *   b = (6 15 4)'
 */

void test_lu1()
{
	using cusolver::lu::solver;
	typedef real64_t value_t;
	constexpr std::size_t N = 3;

	ublas::matrix<value_t> A(N, N);
//	A <<= 2, 0, 2, 1, 2, 2, 0, 2, 3;
	A <<= 1,2,3,4,5,6,2,1,1;

	const solver<value_t> solve(A);

	ublas::vector<value_t> y(N);
//	y <<= 8, 9, 7;
	y <<= 6, 15, 4;

	const ublas::vector<value_t> x = solve(y);

	std::cout << "x = " << x << std::endl;
}

void test_lu2()
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

template <typename T>
class generator
{
public:
	generator()
	:
		_gen(std::random_device()()),
		_dis(1, 2)
	{
	}

	T
	operator()()
	{
		return gen();
	}

	ublas::vector<T>
	operator()(const std::size_t s)
	{
		ublas::vector<T> v(s);
		boost::range::generate(v.data(), std::bind(&generator<T>::gen, this));
		return std::move(v);
	}

	ublas::matrix<T>
	operator()(const std::size_t r, const std::size_t c)
	{
		ublas::matrix<T> A(r, c);
		boost::range::generate(A.data(), std::bind(&generator<T>::gen, this));
		return std::move(A);
	}

protected:
	T gen()
	{
		return T(_dis(_gen), _dis(_gen));
	}

private:
	std::mt19937 _gen;
	std::uniform_real_distribution<typename T::value_type> _dis;
//	std::uniform_real_distribution<T> _dis;
};

template <typename T, typename F>
static void
run(const std::string& name, F function)
{
	std::cout << name << ':' << std::endl;

	const auto t0 = clk_t::now();
	const T result = function();
	const auto t1 = clk_t::now();

	std::cout << "r = " << result << std::endl;
	std::cout << "t = "<< std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms" << std::endl;
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
	typedef complex64_t value_t;

	constexpr std::size_t N = 1000;

	std::cout << (4 * (N * N) + 3 * N) * sizeof(value_t) << std::endl;

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

	run<value_t>("gpu", gpu);
	run<value_t>("cpu", cpu);
}

int
main()
{
	try
	{
		test();
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << std::endl;
	}
	return 0;
}
