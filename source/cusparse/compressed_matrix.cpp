/*
 * compressed_matrix.cpp
 *
 *  Created on: 14.05.2015
 *      Author: mike_gresens
 */

#include <boost/numeric/cusparse/compressed_matrix.hpp>
#include <cuda_runtime.h>

namespace boost {
namespace numeric {
namespace cusparse {

template <typename T>
compressed_matrix<T>::compressed_matrix(const ublas::compressed_matrix<T, ublas::row_major, 0, ublas::unbounded_array<int>>& m)
:
	_size(m.size1()),
	_none_zero(m.nnz()),
	_rows(cuda::make_container<int>(m.index1_data().size())),
	_cols(cuda::make_container<int>(m.index2_data().size())),
	_vals(cuda::make_container<T>(m.value_data().size()))
{
	cudaMemcpy(_rows.get(), m.index1_data().begin(), m.index1_data().size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(_cols.get(), m.index2_data().begin(), m.index2_data().size() * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(_vals.get(), m.value_data().begin(), m.value_data().size() * sizeof(T), cudaMemcpyHostToDevice);
}

template class compressed_matrix<real32_t>;
template class compressed_matrix<real64_t>;
template class compressed_matrix<complex32_t>;
template class compressed_matrix<complex64_t>;

}
}
}
