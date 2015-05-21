/*
 * runner.hpp
 *
 *  Created on: 21.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <string>
#include <chrono>
#include <iostream>

using namespace std::chrono;

class runner
{
public:
	runner(std::string&& name)
	:
		_name(std::forward<std::string>(name))
	{
	}

	template <typename F>
	void operator()(F function) const
	{
		std::cout << _name << ':' << std::endl;

		const auto t0 = system_clock::now();
		const auto result = function();
		const auto t1 = system_clock::now();

		std::cout << "r = " << result << std::endl;
		std::cout << "t = "<< duration_cast<milliseconds>(t1 - t0).count() << " ms" << std::endl;
	}

private:
	std::string _name;
};
