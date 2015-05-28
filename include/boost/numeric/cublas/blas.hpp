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

/// cublas
/// @see http://docs.nvidia.com/cuda/cublas/index.html
namespace cublas {

/// @name level 1
/// @see http://docs.nvidia.com/cuda/cublas/index.html#cublas-level-1-function-reference
/// @{

/**
 * amax
 * @tparam T The value type.
 * @param vector The vector.
 * @return The index.
 * @see http://docs.nvidia.com/cuda/cublas/index.html#cublasi-lt-t-gt-amax
 */
template <typename T>
std::size_t amax(const vector<T>& vector);

/**
 * amin
 * @tparam T The value type.
 * @param vector The vector.
 * @return The index.
 * @see http://docs.nvidia.com/cuda/cublas/index.html#cublasi-lt-t-gt-amin
 */
template <typename T>
std::size_t amin(const vector<T>& vector);

//template <typename T>
//T asum(const vector<T>& vector);

/**
 * Computes \f$y = \alpha * x + y\f$
 * @tparam T The value type.
 * @tparam C The container type.
 * @param a The \f$\alpha\f$.
 * @param x The \f$x\f$.
 * @param y The \f$y\f$.
 * @see http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-axpy
 */
template <typename T, template <typename> class C>
void axpy(const T& a, const C<T>& x, C<T>& y);

/**
 * @tparam T The value type.
 * @tparam C The container type.
 * @param container1 The container1.
 * @param container2 The container2.
 * @see http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-copy
 */
template <typename T, template <typename> class C>
void copy(const C<T>& container1, C<T>& container2);

/**
 * dot
 * @tparam T The value type.
 * @param vector1 The vector1.
 * @param vector2 The vector2.
 * @return The value.
 * @see http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-dot
 */
template <typename T>
T dot(const vector<T>& vector1, const vector<T>& vector2);

/**
 * scal
 * @tparam T The value type.
 * @tparam C The container type.
 * @param alpha The alpha.
 * @param container The container.
 * @see http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-scal
 */
template <typename T, template <typename> class C>
void scal(const T& alpha, C<T>& container);

/// @}

/// @name level 2
/// @see http://docs.nvidia.com/cuda/cublas/index.html#cublas-level-2-function-reference
/// @{

/**
 * gemv
 * @tparam T The value type.
 * @param alpha The alpha.
 * @param matrix The matrix.
 * @param vector1 The vector1.
 * @param beta The beta.
 * @param vector2 The vector2.
 * @see http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemv
 */
template <typename T>
void gemv(const T& alpha, const matrix<T>& matrix, const vector<T>& vector1, const T& beta, vector<T>& vector2);

/// @}

/// @name level 3
/// @see http://docs.nvidia.com/cuda/cublas/index.html#cublas-level-3-function-reference
/// @{

/**
 * gemm
 * @tparam T The value type.
 * @param alpha The alpha.
 * @param matrix1 The matrix1.
 * @param matrix2 The matrix2.
 * @param beta The beta.
 * @param matrix3 The matrix3.
 * @see http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemm
 */
template <typename T>
void gemm(const T& alpha, const matrix<T>& matrix1, const matrix<T>& matrix2, const T& beta, matrix<T>& matrix3);

/**
 * trsm
 * @tparam T The value type.
 * @tparam C The container type.
 * @param alpha The alpha.
 * @param matrix The matrix.
 * @param container The container.
 * @see http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-trsm
 */
template <typename T, template <typename> class C>
void trsm(const T& alpha, const matrix<T>& matrix, C<T>& container);

/// @}

}
}
}
