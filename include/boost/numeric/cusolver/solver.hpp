/*
 * solver.hpp
 *
 *  Created on: 18.05.2015
 *      Author: mike_gresens
 */

#pragma once

#include <boost/numeric/cublas/matrix.hpp>
#include <boost/numeric/cublas/vector.hpp>

namespace boost {
namespace numeric {
namespace cusolver {

/**
 * LU decomposition.
 * @param matrix The matrix.
 * @param pivot The pivot.
 * @see http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-getrf
 */
template <typename T>
void getrf(cublas::matrix<T>& matrix, cuda::container<int>& pivot);

template <typename T>
void getrs(const cublas::matrix<T>& matrix, const cuda::container<int>& pivot, cublas::vector<T>& vector);

template <typename T>
void getrs(const cublas::matrix<T>& matrix, const cuda::container<int>& pivot, cublas::matrix<T>& matrix2);

/**
 * QR decomposition.
 * @param matrix The matrix.
 * @param tau The tau.
 * @return The work.
 * @see http://docs.nvidia.com/cuda/cusolver/index.html#cuds-lt-t-gt-geqrf
 */
template <typename T>
cuda::container<T> geqrf(cublas::matrix<T>& matrix, cuda::container<T>& tau);

template <typename T>
void ormqr(const cublas::matrix<T>& matrix, const cuda::container<T>& tau, cuda::container<T>& work, cublas::vector<T>& vector);

template <typename T>
void ormqr(const cublas::matrix<T>& matrix, const cuda::container<T>& tau, cuda::container<T>& work, cublas::matrix<T>& matrix2);

}
}
}
