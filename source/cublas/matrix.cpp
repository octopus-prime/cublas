/*
 * matrix.cpp
 *
 *  Created on: 14.05.2015
 *      Author: mike_gresens
 */

#include <boost/numeric/cublas/matrix.hpp>
#include <boost/numeric/cublas/blas.hpp>
#include "handle.hpp"
#include "error.hpp"
#include "type_trait.hpp"
#include <cublas_v2.h>

//#define TRANSPOSE_IN_GPU

namespace boost {
namespace numeric {
namespace cublas {

#ifdef TRANSPOSE_IN_GPU

namespace detail {

template <typename U, typename T, typename F>
void transpose(const matrix<T>& matrix1, matrix<T>& matrix2, F function)
{
	const std::size_t m = matrix1.cols();
	const std::size_t n = matrix1.rows();

	constexpr T a = 1, b = 0;
	const cublasStatus_t status = function(handle.get(), CUBLAS_OP_T, CUBLAS_OP_N, m, n, (U*) &a, (U*) (*matrix1).get(), n, (U*) &b, (U*) (*matrix2).get(), m, (U*) (*matrix2).get(), m);
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
}

}

template <typename T>
void transpose(const matrix<T>& matrix1, matrix<T>& matrix2);

template <>
void transpose(const matrix<real32_t>& matrix1, matrix<real32_t>& matrix2)
{
	detail::transpose<float>(matrix1, matrix2, cublasSgeam);
}

template <>
void transpose(const matrix<real64_t>& matrix1, matrix<real64_t>& matrix2)
{
	detail::transpose<double>(matrix1, matrix2, cublasDgeam);
}

template <>
void transpose(const matrix<complex32_t>& matrix1, matrix<complex32_t>& matrix2)
{
	detail::transpose<cuFloatComplex>(matrix1, matrix2, cublasCgeam);
}

template <>
void transpose(const matrix<complex64_t>& matrix1, matrix<complex64_t>& matrix2)
{
	detail::transpose<cuDoubleComplex>(matrix1, matrix2, cublasZgeam);
}

#endif

template <typename T>
matrix<T>::matrix()
:
	_rows(),
	_cols(),
	_elements()
{
}

template <typename T>
matrix<T>::matrix(const std::size_t rows, const std::size_t cols)
:
	_rows(rows),
	_cols(cols),
	_elements(cuda::make_container<T>(_rows * _cols))
{
}


template <typename T>
matrix<T>::matrix(const cublas::matrix<T>& matrix)
:
	_rows(matrix._rows),
	_cols(matrix._cols),
	_elements(cuda::make_container<T>(_rows * _cols))
{
	copy(matrix, *this);
}

//template <typename T>
//matrix<T>::matrix(cublas::matrix<T>&& matrix) noexcept
//:
//	_rows(),
//	_cols(),
//	_elements()
//{
//	std::swap(_rows, matrix._rows);
//	std::swap(_cols, matrix._cols);
//	std::swap(_elements, matrix._elements);
//}

template <typename T>
matrix<T>::matrix(const ublas::matrix<T>& matrix)
:
	_rows(matrix.size1()),
	_cols(matrix.size2()),
	_elements(cuda::make_container<T>(_rows * _cols))
{
#ifdef TRANSPOSE_IN_GPU
	const cublas::matrix<T> temp(_cols, _rows);
	const cublasStatus_t status = cublasSetMatrix(this->rows(), this->cols(), sizeof(T), &matrix.data()[0], this->rows(), (*temp).get(), this->rows());
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
	transpose(temp, *this);
#else
	const ublas::matrix<T> matrix2 = ublas::trans(matrix); // todo: use cublas here !!

	const cublasStatus_t status = cublasSetMatrix(_rows, _cols, sizeof(T), &matrix2.data()[0], _rows, _elements.get(), _rows);
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
#endif
}

template <typename T>
matrix<T>&
matrix<T>::operator=(const cublas::matrix<T>& matrix)
{
	_rows = matrix._rows;
	_cols = matrix._cols;
	_elements = cuda::make_container<T>(_rows * _cols);
	copy(matrix, *this);
	return *this;
}

//template <typename T>
//matrix<T>&
//matrix<T>::operator=(cublas::matrix<T>&& matrix) noexcept
//{
//	_rows = 0;
//	_cols = 0;
//	_elements.release();
//	std::swap(_rows, matrix._rows);
//	std::swap(_cols, matrix._cols);
//	std::swap(_elements, matrix._elements);
//	return *this;
//}

template <typename T>
matrix<T>::operator ublas::matrix<T>() const
{
#ifdef TRANSPOSE_IN_GPU
	cublas::matrix<T> temp(_cols, _rows);
	transpose(*this, temp);

	ublas::matrix<T> matrix(_rows, _cols);
	const cublasStatus_t status = cublasGetMatrix(this->rows(), this->cols(), sizeof(T), (*temp).get(), this->rows(), &matrix.data()[0], this->rows());
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);
	return std::move(matrix);
#else
	ublas::matrix<T> matrix(_cols, _rows);
	const cublasStatus_t status = cublasGetMatrix(_rows, _cols, sizeof(T), _elements.get(), _rows, &matrix.data()[0], _rows);
	if (status != CUBLAS_STATUS_SUCCESS)
		throw std::system_error(status, category, __func__);

	const ublas::matrix<T> matrix2 = ublas::trans(matrix); // todo: use cublas here !!

	return std::move(matrix2);
#endif
}

template <typename T>
matrix<T>&
matrix<T>::operator+=(const matrix<T>& matrix)
{
	axpy(T(+1), matrix, *this);
	return *this;
}

template <typename T>
matrix<T>&
matrix<T>::operator-=(const matrix<T>& matrix)
{
	axpy(T(-1), matrix, *this);
	return *this;
}

template <typename T>
matrix<T>&
matrix<T>::operator*=(const T& value)
{
	scal(value, *this);
	return *this;
}

template <typename T>
matrix<T>&
matrix<T>::operator/=(const T& value)
{
	scal(T(1) / value, *this);
	return *this;
}

template <typename T>
matrix<T>
operator*(const matrix<T>& matrix1, const matrix<T>& matrix2)
{
	matrix<T> result(matrix1.rows(), matrix2.cols());
	gemm(T(1), matrix1, matrix2, T(0), result);
	return std::move(result);
}

template class matrix<real32_t>;
template class matrix<real64_t>;
template class matrix<complex32_t>;
template class matrix<complex64_t>;

template matrix<real32_t> operator*(const matrix<real32_t>& matrix1, const matrix<real32_t>& matrix2);
template matrix<real64_t> operator*(const matrix<real64_t>& matrix1, const matrix<real64_t>& matrix2);
template matrix<complex32_t> operator*(const matrix<complex32_t>& matrix1, const matrix<complex32_t>& matrix2);
template matrix<complex64_t> operator*(const matrix<complex64_t>& matrix1, const matrix<complex64_t>& matrix2);

}
}
}
