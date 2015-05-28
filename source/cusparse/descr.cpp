/*
 * descr.cpp
 *
 *  Created on: 28.05.2015
 *      Author: mike_gresens
 */

#include "descr.hpp"

namespace boost {
namespace numeric {
namespace cusparse {

static descr_t
make_descr()
{
	cusparseMatDescr_t desc = nullptr;
	const cusparseStatus_t status = cusparseCreateMatDescr(&desc);
	if (status != CUSPARSE_STATUS_SUCCESS)
		throw std::runtime_error("cusparseCreateMatDescr");
    cusparseSetMatType(desc, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(desc, CUSPARSE_INDEX_BASE_ZERO);
	return std::shared_ptr<cusparseMatDescr>(desc, cusparseDestroyMatDescr);
}

const descr_t descr = make_descr();

}
}
}
