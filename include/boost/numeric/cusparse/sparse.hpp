/*
 * solver.hpp
 *
 *  Created on: 18.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <boost/numeric/cusparse/compressed_matrix.hpp>
#include <boost/numeric/cublas/vector.hpp>

namespace boost {
namespace numeric {
namespace cusparse {

template <typename T>
void csrlsvqr(const compressed_matrix<T>& A, const cublas::vector<T>& b, cublas::vector<T>& x);

template <typename T>
void csreigvsi(const compressed_matrix<T>& A, const T& mu0, const cublas::vector<T>& x0, const std::size_t max, T& mu, cublas::vector<T>& x);

}
}
}
