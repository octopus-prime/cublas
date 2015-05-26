/*
 * size.hpp
 *
 *  Created on: 26.05.2015
 *      Author: mike_gresens
 */

#pragma once

namespace boost {
namespace numeric {
namespace cublas {

template <typename T>
std::size_t size(const vector<T>& vector)
{
	return vector.size();
}

template <typename T>
std::size_t size(const matrix<T>& matrix)
{
	return matrix.rows() * matrix.cols();
}

template <typename T>
std::size_t rows(const vector<T>& vector)
{
	return vector.size();
}

template <typename T>
std::size_t rows(const matrix<T>& matrix)
{
	return matrix.rows();
}

template <typename T>
std::size_t cols(const vector<T>& vector)
{
	return 1;
}

template <typename T>
std::size_t cols(const matrix<T>& matrix)
{
	return matrix.cols();
}

}
}
}
