#ifndef _CSR_CUSP_SPTRSV_H_
#define _CSR_CUSP_SPTRSV_H_

#include "common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#include <cusparse.h>
#if defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION < 12000)
#include <cusparse_v2.h>
#endif

// -------------------------
// error check helpers (return int error code)
// -------------------------
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do {                                                     \
    cudaError_t _e = (call);                                                      \
    if (_e != cudaSuccess) {                                                      \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,              \
                cudaGetErrorString(_e));                                          \
        return -1;                                                                \
    }                                                                             \
} while(0)
#endif

#ifndef CUSPARSE_CHECK
#define CUSPARSE_CHECK(call) do {                                                 \
    cusparseStatus_t _s = (call);                                                 \
    if (_s != CUSPARSE_STATUS_SUCCESS) {                                          \
        fprintf(stderr, "cuSPARSE error %s:%d: status=%d\n", __FILE__, __LINE__,   \
                (int)_s);                                                         \
        return -1;                                                                \
    }                                                                             \
} while(0)
#endif

// -------------------------
// Simple dispatch for float/double
// You MUST keep VALUE_TYPE consistent with USE_DOUBLE.
//   float : (default)            VALUE_TYPE=float,  no -DUSE_DOUBLE
//   double: compile with         -DUSE_DOUBLE -DVALUE_TYPE=double
// -------------------------
#ifdef USE_DOUBLE
  #define CSRsv2_bufferSize  cusparseDcsrsv2_bufferSize
  #define CSRsv2_analysis    cusparseDcsrsv2_analysis
  #define CSRsv2_solve       cusparseDcsrsv2_solve
  #define BSRI_CUDA_VALUE_T  CUDA_R_64F
#else
  #define CSRsv2_bufferSize  cusparseScsrsv2_bufferSize
  #define CSRsv2_analysis    cusparseScsrsv2_analysis
  #define CSRsv2_solve       cusparseScsrsv2_solve
  #define BSRI_CUDA_VALUE_T  CUDA_R_32F
#endif

static inline int csr_cuSP_SpTRSV(
    int m,
    const int* csrRowPtrA,
    const int* csrColIdxA,
    const VALUE_TYPE* csrValA,
    int nnz,
    VALUE_TYPE* b
) {
    // --------------------
    // H2D
    // --------------------
    VALUE_TYPE *d_csrValA = nullptr;
    int *d_csrRowPtrA = nullptr;
    int *d_csrColIdxA = nullptr;

    VALUE_TYPE *d_x = nullptr;  // b
    VALUE_TYPE *d_y = nullptr;  // x
    VALUE_TYPE *d_z = nullptr;  // intermediate

    CUDA_CHECK(cudaMalloc((void**)&d_csrValA,    (size_t)nnz * sizeof(VALUE_TYPE)));
    CUDA_CHECK(cudaMalloc((void**)&d_csrRowPtrA, (size_t)(m + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_csrColIdxA, (size_t)nnz * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_csrValA,    csrValA,    (size_t)nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (size_t)(m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_csrColIdxA, csrColIdxA, (size_t)nnz * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMalloc((void**)&d_x, (size_t)m * sizeof(VALUE_TYPE)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, (size_t)m * sizeof(VALUE_TYPE)));
    CUDA_CHECK(cudaMalloc((void**)&d_z, (size_t)m * sizeof(VALUE_TYPE)));

    CUDA_CHECK(cudaMemcpy(d_x, b, (size_t)m * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, (size_t)m * sizeof(VALUE_TYPE)));
    CUDA_CHECK(cudaMemset(d_z, 0, (size_t)m * sizeof(VALUE_TYPE)));

    // --------------------
    // cuSPARSE init
    // --------------------
    cusparseHandle_t handle = nullptr;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    struct timeval t1, t2;
    const VALUE_TYPE alpha = static_cast<VALUE_TYPE>(1.0);

#if defined(CUSPARSE_VERSION) && (CUSPARSE_VERSION >= 12000)
    // =========================================================
    // CUDA 12+ : Generic API cusparseSpSV
    // =========================================================
    cusparseSpMatDescr_t matL = nullptr, matU = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecZ = nullptr, vecY = nullptr;
    cusparseSpSVDescr_t  spsvL = nullptr, spsvU = nullptr;

    const cusparseIndexType_t itype = CUSPARSE_INDEX_32I;
    const cusparseIndexBase_t ibase = CUSPARSE_INDEX_BASE_ZERO;

    // FIX: valueType dispatch by VALUE_TYPE
    const cudaDataType valueType = BSRI_CUDA_VALUE_T;

    CUSPARSE_CHECK(cusparseCreateCsr(
        &matL, m, m, nnz,
        (void*)d_csrRowPtrA, (void*)d_csrColIdxA, (void*)d_csrValA,
        itype, itype, ibase, valueType));

    CUSPARSE_CHECK(cusparseCreateCsr(
        &matU, m, m, nnz,
        (void*)d_csrRowPtrA, (void*)d_csrColIdxA, (void*)d_csrValA,
        itype, itype, ibase, valueType));

    // triangular attributes
    {
        cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER;
        cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_UNIT;
        CUSPARSE_CHECK(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill)));
        CUSPARSE_CHECK(cusparseSpMatSetAttribute(matL, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));
    }
    {
        cusparseFillMode_t fill = CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT;
        CUSPARSE_CHECK(cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill)));
        CUSPARSE_CHECK(cusparseSpMatSetAttribute(matU, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag)));
    }

    CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, m, (void*)d_x, valueType));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecZ, m, (void*)d_z, valueType));
    CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, m, (void*)d_y, valueType));

    CUSPARSE_CHECK(cusparseSpSV_createDescr(&spsvL));
    CUSPARSE_CHECK(cusparseSpSV_createDescr(&spsvU));

    size_t bsL = 0, bsU = 0;
    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matL, vecX, vecZ,
        valueType, CUSPARSE_SPSV_ALG_DEFAULT, spsvL, &bsL));

    CUSPARSE_CHECK(cusparseSpSV_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matU, vecZ, vecY,
        valueType, CUSPARSE_SPSV_ALG_DEFAULT, spsvU, &bsU));

    size_t bs = (bsL > bsU) ? bsL : bsU;
    void* dBuffer = nullptr;
    if (bs > 0) CUDA_CHECK(cudaMalloc(&dBuffer, bs));

    CUSPARSE_CHECK(cusparseSpSV_analysis(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matL, vecX, vecZ,
        valueType, CUSPARSE_SPSV_ALG_DEFAULT, spsvL, dBuffer));

    CUSPARSE_CHECK(cusparseSpSV_analysis(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matU, vecZ, vecY,
        valueType, CUSPARSE_SPSV_ALG_DEFAULT, spsvU, dBuffer));

    gettimeofday(&t1, NULL);

    CUSPARSE_CHECK(cusparseSpSV_solve(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matL, vecX, vecZ,
        valueType, CUSPARSE_SPSV_ALG_DEFAULT, spsvL));

    CUSPARSE_CHECK(cusparseSpSV_solve(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, matU, vecZ, vecY,
        valueType, CUSPARSE_SPSV_ALG_DEFAULT, spsvU));

    CUDA_CHECK(cudaDeviceSynchronize());
    gettimeofday(&t2, NULL);

    VALUE_TYPE time_ms = static_cast<VALUE_TYPE>((t2.tv_sec - t1.tv_sec) * 1000.0 +
                                                 (t2.tv_usec - t1.tv_usec) / 1000.0);
    printf("\ncuSPARSE SpSV (CUDA12+) time: %f ms\n", (double)time_ms);

    if (dBuffer) CUDA_CHECK(cudaFree(dBuffer));
    CUSPARSE_CHECK(cusparseSpSV_destroyDescr(spsvL));
    CUSPARSE_CHECK(cusparseSpSV_destroyDescr(spsvU));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecZ));
    CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    CUSPARSE_CHECK(cusparseDestroySpMat(matL));
    CUSPARSE_CHECK(cusparseDestroySpMat(matU));

#else
    // =========================================================
    // CUDA 11.x : legacy csrsv2
    // =========================================================
    cusparseMatDescr_t descr_L = nullptr;
    cusparseMatDescr_t descr_U = nullptr;
    csrsv2Info_t info_L = 0;
    csrsv2Info_t info_U = 0;

    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

    CUSPARSE_CHECK(cusparseCreateCsrsv2Info(&info_L));
    CUSPARSE_CHECK(cusparseCreateCsrsv2Info(&info_U));

    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr_L));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER));
    CUSPARSE_CHECK(cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT));

    CUSPARSE_CHECK(cusparseCreateMatDescr(&descr_U));
    CUSPARSE_CHECK(cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO));
    CUSPARSE_CHECK(cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL));
    CUSPARSE_CHECK(cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER));
    CUSPARSE_CHECK(cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT));

    int bsL = 0, bsU = 0;
    CUSPARSE_CHECK(CSRsv2_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, nnz, descr_L, d_csrValA, d_csrRowPtrA, d_csrColIdxA,
        info_L, &bsL));

    CUSPARSE_CHECK(CSRsv2_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, nnz, descr_U, d_csrValA, d_csrRowPtrA, d_csrColIdxA,
        info_U, &bsU));

    int bs = (bsL > bsU) ? bsL : bsU;
    void* dBuffer = nullptr;
    CUDA_CHECK(cudaMalloc(&dBuffer, (size_t)bs));

    CUSPARSE_CHECK(CSRsv2_analysis(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, nnz, descr_L, d_csrValA, d_csrRowPtrA, d_csrColIdxA,
        info_L, policy_L, dBuffer));

    CUSPARSE_CHECK(CSRsv2_analysis(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, nnz, descr_U, d_csrValA, d_csrRowPtrA, d_csrColIdxA,
        info_U, policy_U, dBuffer));

    gettimeofday(&t1, NULL);

    CUSPARSE_CHECK(CSRsv2_solve(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, nnz, &alpha, descr_L, d_csrValA, d_csrRowPtrA, d_csrColIdxA,
        info_L, d_x, d_z, policy_L, dBuffer));

    CUSPARSE_CHECK(CSRsv2_solve(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        m, nnz, &alpha, descr_U, d_csrValA, d_csrRowPtrA, d_csrColIdxA,
        info_U, d_z, d_y, policy_U, dBuffer));

    CUDA_CHECK(cudaDeviceSynchronize());
    gettimeofday(&t2, NULL);

    VALUE_TYPE time_ms = static_cast<VALUE_TYPE>((t2.tv_sec - t1.tv_sec) * 1000.0 +
                                                 (t2.tv_usec - t1.tv_usec) / 1000.0);
    printf("\ncuSPARSE csrsv2 (CUDA11) time: %f ms\n", (double)time_ms);

    CUDA_CHECK(cudaFree(dBuffer));
    CUSPARSE_CHECK(cusparseDestroyCsrsv2Info(info_L));
    CUSPARSE_CHECK(cusparseDestroyCsrsv2Info(info_U));
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr_L));
    CUSPARSE_CHECK(cusparseDestroyMatDescr(descr_U));
#endif

    // D2H result
    CUDA_CHECK(cudaMemcpy(b, d_y, (size_t)m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

    // common cleanup
    CUSPARSE_CHECK(cusparseDestroy(handle));

    CUDA_CHECK(cudaFree(d_csrValA));
    CUDA_CHECK(cudaFree(d_csrRowPtrA));
    CUDA_CHECK(cudaFree(d_csrColIdxA));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    return 0;
}

// optional: avoid leaking macros
#undef CSRsv2_bufferSize
#undef CSRsv2_analysis
#undef CSRsv2_solve
#undef BSRI_CUDA_VALUE_T

#endif // _CSR_CUSP_SPTRSV_H_
