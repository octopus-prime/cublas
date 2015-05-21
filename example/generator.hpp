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
