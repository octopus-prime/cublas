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
void scal(const T& alpha, cuda::container<T>& container);

template <typename T>
void axpy(const T& alpha, const cuda::container<T>& container1, cuda::container<T>& container2);

template <typename T>
T dot(const vector<T>& vector1, const vector<T>& vector2);

// blas2
template <typename T>
void gemv(const T& alpha, const matrix<T>& matrix, const vector<T>& vector1, const T& beta, vector<T>& vector2);

// blas3
template <typename T>
void gemm(const T& alpha, const matrix<T>& matrix1, const matrix<T>& matrix2, const T& beta, matrix<T>& matrix3);

template <typename T>
void trsm(const T& alpha, const matrix<T>& matrix, vector<T>& vector);

template <typename T>
void trsm(const T& alpha, const matrix<T>& matrix1, matrix<T>& matrix2);

}
}
}
