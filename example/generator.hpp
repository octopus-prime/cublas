/*
 * generator.hpp
 *
 *  Created on: 21.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/range/algorithm/generate.hpp>
#include <random>

using namespace boost::numeric;

template <typename T>
class generator
{
	struct real_generator
	{
		template <typename F>
		T operator()(F function) const
		{
			return function();
		}
	};

	struct complex_generator
	{
		template <typename F>
		T operator()(F function) const
		{
			return T(function(), function());
		}
	};

public:
	generator()
	:
		_gen(std::random_device()()),
		_dis(0.1, 0.9)
	{
	}

	T
	operator()()
	{
		return generate();
	}

	ublas::vector<T>
	operator()(const std::size_t s)
	{
		ublas::vector<T> v(s);
		boost::range::generate(v.data(), std::bind(&generator<T>::generate<>, this));
		return std::move(v);
	}

	ublas::matrix<T>
	operator()(const std::size_t r, const std::size_t c)
	{
		ublas::matrix<T> A(r, c);
		boost::range::generate(A.data(), std::bind(&generator<T>::generate<>, this));
		return std::move(A);
	}

protected:
	template <typename G = typename std::conditional<std::is_floating_point<T>::value, real_generator, complex_generator>::type>
	T generate()
	{
		return G()(std::bind(&generator<T>::random, this));
	}

	double random()
	{
		return _dis(_gen);
	}

private:
	std::mt19937 _gen;
	std::uniform_real_distribution<double> _dis;
};
