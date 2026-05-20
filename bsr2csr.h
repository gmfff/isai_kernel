#include "common.h"
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <assert.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef CHECK_CUDA
#define CHECK_CUDA(call) do{ auto _e=(call); if(_e!=cudaSuccess){ \
  printf("CUDA error %s @%s:%d\n", cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} }while(0)
#endif

#ifndef CHECK_CUSPARSE
#define CHECK_CUSPARSE(call) do{ auto _s=(call); if(_s!=CUSPARSE_STATUS_SUCCESS){ \
  printf("cuSPARSE error %d @%s:%d\n", (int)_s, __FILE__, __LINE__); exit(1);} }while(0)
#endif

int bsr2csr_cusparse(const int *bsrRowPtr_A,
                     const int *bsrColInd_A,
                     const VALUE_TYPE *bsrVal_A,
                     const int mb,
                     const int nnzb,
                     const int blockDim,
                     int **bsr2csrRowPtrA_tmp,
                     int **bsr2csrColIndA_tmp,
                     VALUE_TYPE **bsr2csrVal_tmp,
                     int *nnzbsr2csr)
{
    int *d_csrRowPtrAAA = nullptr;
    int *d_csrColIndAAA = nullptr;
    VALUE_TYPE *d_csrValAAA = nullptr;

    int *d_bsrRowPtrAAA = nullptr;
    int *d_bsrColIndAAA = nullptr;
    VALUE_TYPE *d_bsrValAAA = nullptr;

    int *h_csrRowPtrAAA = nullptr;
    int *h_csrColIndAAA = nullptr;
    VALUE_TYPE *h_csrValAAA = nullptr;

    int m   = mb * blockDim;
    int nnz = nnzb * blockDim * blockDim;  // number of scalar elements in CSR

    struct timeval t1, t2;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&d_csrRowPtrAAA, sizeof(int) * (m + 1)));
    CHECK_CUDA(cudaMalloc((void**)&d_csrColIndAAA, sizeof(int) * nnz));
    CHECK_CUDA(cudaMalloc((void**)&d_csrValAAA, sizeof(VALUE_TYPE) * nnz));

    CHECK_CUDA(cudaMalloc((void**)&d_bsrRowPtrAAA, sizeof(int) * (mb + 1)));
    CHECK_CUDA(cudaMalloc((void**)&d_bsrColIndAAA, sizeof(int) * nnzb));
    CHECK_CUDA(cudaMalloc((void**)&d_bsrValAAA, sizeof(VALUE_TYPE) * nnz));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_bsrRowPtrAAA, bsrRowPtr_A, sizeof(int) * (mb + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bsrColIndAAA, bsrColInd_A, sizeof(int) * nnzb, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bsrValAAA,   bsrVal_A,    sizeof(VALUE_TYPE) * nnz, cudaMemcpyHostToDevice));

    // Allocate host memory
    h_csrRowPtrAAA = (int*)malloc((m + 1) * sizeof(int));
    h_csrColIndAAA = (int*)malloc(nnz * sizeof(int));
    h_csrValAAA    = (VALUE_TYPE*)malloc(nnz * sizeof(VALUE_TYPE));

    // Create cuSPARSE handle and matrix descriptors
    cusparseHandle_t handle = 0;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;

    cusparseMatDescr_t descr = 0;
    cusparseMatDescr_t descr_bsr = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL));

    CHECK_CUSPARSE(cusparseCreateMatDescr(&descr_bsr));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descr_bsr, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descr_bsr, CUSPARSE_MATRIX_TYPE_GENERAL));

    CHECK_CUDA(cudaDeviceSynchronize());
    gettimeofday(&t1, NULL);

    // Convert BSR to CSR format (dispatch by USE_DOUBLE)
#ifdef USE_DOUBLE
    CHECK_CUSPARSE(cusparseDbsr2csr(
        handle, dir,
        mb, mb,
        descr_bsr,
        (const double*)d_bsrValAAA, d_bsrRowPtrAAA, d_bsrColIndAAA,
        blockDim,
        descr,
        (double*)d_csrValAAA, d_csrRowPtrAAA, d_csrColIndAAA));
#else
    CHECK_CUSPARSE(cusparseSbsr2csr(
        handle, dir,
        mb, mb,
        descr_bsr,
        (const float*)d_bsrValAAA, d_bsrRowPtrAAA, d_bsrColIndAAA,
        blockDim,
        descr,
        (float*)d_csrValAAA, d_csrRowPtrAAA, d_csrColIndAAA));
#endif

    CHECK_CUDA(cudaDeviceSynchronize());
    gettimeofday(&t2, NULL);

    double time_ms = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("bsr2csr time: %f ms\n", time_ms);

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_csrValAAA,    d_csrValAAA,    sizeof(VALUE_TYPE) * nnz, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_csrRowPtrAAA, d_csrRowPtrAAA, sizeof(int) * (m + 1),   cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_csrColIndAAA, d_csrColIndAAA, sizeof(int) * nnz,       cudaMemcpyDeviceToHost));

    *bsr2csrRowPtrA_tmp = h_csrRowPtrAAA;
    *bsr2csrColIndA_tmp = h_csrColIndAAA;
    *bsr2csrVal_tmp     = h_csrValAAA;
    *nnzbsr2csr          = nnz;

    // Cleanup device
    cudaFree(d_csrRowPtrAAA);
    cudaFree(d_csrColIndAAA);
    cudaFree(d_csrValAAA);
    cudaFree(d_bsrRowPtrAAA);
    cudaFree(d_bsrColIndAAA);
    cudaFree(d_bsrValAAA);

    // Cleanup cusparse
    cusparseDestroyMatDescr(descr);
    cusparseDestroyMatDescr(descr_bsr);
    cusparseDestroy(handle);

    // host memory is returned to caller (so do NOT free here)
    return 0;
}
