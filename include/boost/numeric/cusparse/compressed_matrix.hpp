/*
 * compressed_matrix.hpp
 *
 *  Created on: 14.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <boost/numeric/cuda/container.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

namespace boost {
namespace numeric {
namespace cusparse {

template <typename T>
class compressed_matrix
{
public:
	compressed_matrix(const ublas::compressed_matrix<T, ublas::row_major, 0, ublas::unbounded_array<int>>& m);

	const cuda::container<int>& cols() const noexcept
	{
		return _cols;
	}

	std::size_t nonezero() const noexcept
	{
		return _none_zero;
	}

	const cuda::container<int>& rows() const noexcept
	{
		return _rows;
	}

	std::size_t size() const  noexcept
	{
		return _size;
	}

	const cuda::container<T>& vals() const  noexcept
	{
		return _vals;
	}

private:
	std::size_t _size;
	std::size_t _none_zero;
    cuda::container<int> _rows;
    cuda::container<int> _cols;
    cuda::container<T> _vals;
};

}
}
}
