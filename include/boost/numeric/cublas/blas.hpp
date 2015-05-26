/*
 * blas.hpp
 *
 *  Created on: 09.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <boost/numeric/cublas/matrix.hpp>
#include <boost/numeric/cublas/vector.hpp>

namespace boost {
namespace numeric {
namespace cublas {

// blas1
template <typename T>
std::size_t amax(const vector<T>& vector);

template <typename T>
std::size_t amin(const vector<T>& vector);

//template <typename T>
//T asum(const vector<T>& vector);

template <typename T, template <typename> class C>
void axpy(const T& alpha, const C<T>& container1, C<T>& container2);

template <typename T, template <typename> class C>
void scal(const T& alpha, C<T>& container);

/**
 * @tparam T The value type, e.g. real32_t or complex64_t.
 * @tparam C The container type, e.g. vector or matrix.
 * @param container1 The 1st container.
 * @param container2 The 2nd container.
 */
template <typename T, template <typename> class C>
void copy(const C<T>& container1, C<T>& container2);

template <typename T>
T dot(const vector<T>& vector1, const vector<T>& vector2);

// blas2
template <typename T>
void gemv(const T& alpha, const matrix<T>& matrix, const vector<T>& vector1, const T& beta, vector<T>& vector2);

// blas3
template <typename T>
void gemm(const T& alpha, const matrix<T>& matrix1, const matrix<T>& matrix2, const T& beta, matrix<T>& matrix3);

template <typename T, template <typename> class C>
void trsm(const T& alpha, const matrix<T>& matrix, C<T>& container);

}
}
}
