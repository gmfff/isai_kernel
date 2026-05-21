#include "common.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cusparse_v2.h>
#include <assert.h>

#include "bsr2csr.h"
#include "csr_cusp_sptrsv.h"
#include "cape_sptrsv.h"
#include "read_matrix.h"

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

// -------------------------
// simple dispatch for float/double (USE_DOUBLE)
// -------------------------
#ifdef USE_DOUBLE
  #define BSRI_bsrilu02_bufferSize  cusparseDbsrilu02_bufferSize
  #define BSRI_bsrilu02_analysis    cusparseDbsrilu02_analysis
  #define BSRI_bsrilu02_factorize   cusparseDbsrilu02

  #define BSRI_bsrsv2_bufferSize    cusparseDbsrsv2_bufferSize
  #define BSRI_bsrsv2_analysis      cusparseDbsrsv2_analysis
  #define BSRI_bsrsv2_solve         cusparseDbsrsv2_solve
#else
  #define BSRI_bsrilu02_bufferSize  cusparseSbsrilu02_bufferSize
  #define BSRI_bsrilu02_analysis    cusparseSbsrilu02_analysis
  #define BSRI_bsrilu02_factorize   cusparseSbsrilu02

  #define BSRI_bsrsv2_bufferSize    cusparseSbsrsv2_bufferSize
  #define BSRI_bsrsv2_analysis      cusparseSbsrsv2_analysis
  #define BSRI_bsrsv2_solve         cusparseSbsrsv2_solve
#endif

// 你原代码用的 max(...) 没包含 <algorithm>，这里给一个小宏避免改太多
#ifndef BSRI_MAX
#define BSRI_MAX(a,b) (( (a) > (b) ) ? (a) : (b))
#endif

int BILU_cuSPARSE(const int           *bsrRowPtr_A,
                  const int           *bsrColIdx_A,
                  const VALUE_TYPE    *bsrVal_A,
                  const int            m,
                  const int            mb,
                  const int            nnzb,
                  const int            blockDim,
                  VALUE_TYPE          **x_cusp,
                  VALUE_TYPE          **ilu_cusp)
{
    VALUE_TYPE *x_ref; // 未使用，保留你原变量
    (void)x_ref;

    VALUE_TYPE *b = nullptr;
    struct timeval t1, t2;

    ilu_b(m, &b);

    VALUE_TYPE *x      = (VALUE_TYPE*)malloc(sizeof(VALUE_TYPE) * m);
    VALUE_TYPE *ILUval = (VALUE_TYPE*)malloc(sizeof(VALUE_TYPE) * nnzb * blockDim * blockDim);
    int *ILUcolInd     = (int*)malloc(sizeof(int) * nnzb);
    int *ILUrowPtr     = (int*)malloc(sizeof(int) * (mb + 1));

    cusparseHandle_t handle = 0;
    cusparseStatus_t CUIFSUCCESS = cusparseCreate(&handle);
    (void)CUIFSUCCESS;

    int *d_bsrRowPtr = nullptr;
    int *d_bsrColInd = nullptr;
    VALUE_TYPE *d_bsrVal = nullptr;

    VALUE_TYPE *d_x = nullptr;
    VALUE_TYPE *d_y = nullptr;
    VALUE_TYPE *d_z = nullptr;

    cudaMalloc((void**)&d_bsrRowPtr, (mb + 1) * sizeof(int));
    cudaMalloc((void**)&d_bsrColInd, nnzb * sizeof(int));
    cudaMalloc((void**)&d_bsrVal,    (nnzb * blockDim * blockDim) * sizeof(VALUE_TYPE));

    cudaMemcpy(d_bsrRowPtr, bsrRowPtr_A, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bsrColInd, bsrColIdx_A, nnzb * sizeof(int),     cudaMemcpyHostToDevice);
    cudaMemcpy(d_bsrVal,    bsrVal_A,    (nnzb * blockDim * blockDim) * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    // Vector d_x = b
    cudaMalloc((void**)&d_x, m * sizeof(VALUE_TYPE));
    cudaMemcpy(d_x, b, m * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    // Vector d_y
    cudaMalloc((void**)&d_y, m * sizeof(VALUE_TYPE));
    cudaMemset(d_y, 0, m * sizeof(VALUE_TYPE));

    // Vector d_z
    cudaMalloc((void**)&d_z, m * sizeof(VALUE_TYPE));
    cudaMemset(d_z, 0, m * sizeof(VALUE_TYPE));

    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;
    cusparseMatDescr_t descr_U = 0;
    bsrilu02Info_t info_M = 0;
    bsrsv2Info_t info_L = 0;
    bsrsv2Info_t info_U = 0;

    int pBufferSize_M = 0;
    int pBufferSize_L = 0;
    int pBufferSize_U = 0;
    int pBufferSize   = 0;
    void *pBuffer     = 0;

    int structural_zero = -1;
    int numerical_zero  = -1;

    const VALUE_TYPE alpha = (VALUE_TYPE)1.0;

    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

    const cusparseOperation_t trans_L = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U = CUSPARSE_OPERATION_NON_TRANSPOSE;

    const cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    // step 1: create descriptors
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create info
    cusparseCreateBsrilu02Info(&info_M);
    cusparseCreateBsrsv2Info(&info_L);
    cusparseCreateBsrsv2Info(&info_U);

    // step 3: query buffer sizes
    BSRI_bsrilu02_bufferSize(handle, dir, mb, nnzb,
        descr_M, d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim, info_M, &pBufferSize_M);

    BSRI_bsrsv2_bufferSize(handle, dir, trans_L, mb, nnzb,
        descr_L, d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim, info_L, &pBufferSize_L);

    BSRI_bsrsv2_bufferSize(handle, dir, trans_U, mb, nnzb,
        descr_U, d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim, info_U, &pBufferSize_U);

    pBufferSize = BSRI_MAX(pBufferSize_M, BSRI_MAX(pBufferSize_L, pBufferSize_U));

    cudaMalloc((void**)&pBuffer, pBufferSize);

    // analysis
    BSRI_bsrilu02_analysis(handle, dir, mb, nnzb, descr_M,
        d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim, info_M,
        policy_M, pBuffer);

    cusparseStatus_t status = cusparseXbsrilu02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status) {
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    BSRI_bsrsv2_analysis(handle, dir, trans_L, mb, nnzb, descr_L,
        d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim,
        info_L, policy_L, pBuffer);

    BSRI_bsrsv2_analysis(handle, dir, trans_U, mb, nnzb, descr_U,
        d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim,
        info_U, policy_U, pBuffer);

    // step 5: ILU factorization M = L * U (in-place)
    gettimeofday(&t1, NULL);

    BSRI_bsrilu02_factorize(handle, dir, mb, nnzb, descr_M,
        d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim, info_M, policy_M, pBuffer);

    cusparseStatus_t status1 = cusparseXbsrilu02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status1) {
        printf("block U(%d,%d) is not invertible\n", numerical_zero, numerical_zero);
    }

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    double time_cuda_LU = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0);
    printf("\ncuSPARSE block ILU time: %f\n", time_cuda_LU);

    // step 6/7: triangular solves
    //warmup
    for (int i = 0; i < 100; ++i)
    {
        BSRI_bsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L,
        d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim, info_L,
        d_x, d_z, policy_L, pBuffer);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);

    // solve L*z = x
    for (int i = 0; i < 1000; ++i)
    {
        BSRI_bsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L,
            d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim, info_L,
            d_x, d_z, policy_L, pBuffer);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    // solve U*y = z
    BSRI_bsrsv2_solve(handle, dir, trans_U, mb, nnzb, &alpha, descr_U,
        d_bsrVal, d_bsrRowPtr, d_bsrColInd, blockDim, info_U,
        d_z, d_y, policy_U, pBuffer);


    double time_cuda_SV = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0)/1000;
    printf("cuSPARSE block SpTRSV time: %f\n", time_cuda_SV);

    // Copy results back
    cudaMemcpy(x, d_y, m * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(ILUval, d_bsrVal, (nnzb * blockDim * blockDim) * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost);
    cudaMemcpy(ILUcolInd, d_bsrColInd, nnzb * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(ILUrowPtr, d_bsrRowPtr, (mb + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    *x_cusp  = x;
    *ilu_cusp = ILUval;

    // free resources
    cudaFree(pBuffer);

    cusparseDestroyMatDescr(descr_M);
    cusparseDestroyMatDescr(descr_L);
    cusparseDestroyMatDescr(descr_U);
    cusparseDestroyBsrilu02Info(info_M);
    cusparseDestroyBsrsv2Info(info_L);
    cusparseDestroyBsrsv2Info(info_U);
    cusparseDestroy(handle);

    // NOTE: 你原代码这里没 free d_*，会泄露；我按你原逻辑保留，但建议你加上。
    //       如果你确定后续不再用，就应该释放：
    // cudaFree(d_bsrRowPtr); cudaFree(d_bsrColInd); cudaFree(d_bsrVal);
    // cudaFree(d_x); cudaFree(d_y); cudaFree(d_z);

    // pipeline: BSR(ILU) -> CSR -> SpTRSV
    int *bsr2csrRowPtrA = nullptr;
    int *bsr2csrColIdxA = nullptr;
    VALUE_TYPE *bsr2csrValA = nullptr;   // FIX: was double*, must be VALUE_TYPE*
    int nnzbsr2csr_tmp = 0;

    bsr2csr_cusparse(ILUrowPtr, ILUcolInd, ILUval,
                     mb, nnzb, blockDim,
                     &bsr2csrRowPtrA, &bsr2csrColIdxA, &bsr2csrValA,
                     &nnzbsr2csr_tmp);

    csr_cuSP_SpTRSV(m, bsr2csrRowPtrA, bsr2csrColIdxA, bsr2csrValA, nnzbsr2csr_tmp, b);
    csr_cape_SpTRSV(m, bsr2csrRowPtrA, bsr2csrColIdxA, bsr2csrValA, nnzbsr2csr_tmp, b);

    return 0;
}

// optional: avoid leaking macros
#undef BSRI_bsrilu02_bufferSize
#undef BSRI_bsrilu02_analysis
#undef BSRI_bsrilu02_factorize
#undef BSRI_bsrsv2_bufferSize
#undef BSRI_bsrsv2_analysis
#undef BSRI_bsrsv2_solve
