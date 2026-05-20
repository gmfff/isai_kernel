#pragma once
#include "common.h"

#include <cusparse_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#ifndef BSRI_CHECK_CUDA
#define BSRI_CHECK_CUDA(call) do{ auto _e=(call); if(_e!=cudaSuccess){ \
  std::cerr<<"CUDA "<<cudaGetErrorString(_e)<<" @"<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} }while(0)
#endif
#ifndef BSRI_CHECK_CUSPARSE
#define BSRI_CHECK_CUSPARSE(call) do{ auto _s=(call); if(_s!=CUSPARSE_STATUS_SUCCESS){ \
  std::cerr<<"cuSPARSE "<<int(_s)<<" @"<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} }while(0)
#endif

#ifdef USE_DOUBLE
// ======================== double version ========================
inline void bsrilu02_inplace(
    int nb, int nnzb, int BS,
    const int* d_bsrRow, const int* d_bsrCol,
    double* d_bsrVal)
{
    cusparseHandle_t spH; BSRI_CHECK_CUSPARSE(cusparseCreate(&spH));
    cusparseMatDescr_t descrA; BSRI_CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    BSRI_CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    BSRI_CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    bsrilu02Info_t infoM = 0; BSRI_CHECK_CUSPARSE(cusparseCreateBsrilu02Info(&infoM));

    int pBufSize = 0;
    BSRI_CHECK_CUSPARSE(cusparseDbsrilu02_bufferSize(
        spH, CUSPARSE_DIRECTION_ROW, nb, nnzb,
        descrA, d_bsrVal, d_bsrRow, d_bsrCol, BS, infoM, &pBufSize));

    void* pBuf = nullptr; BSRI_CHECK_CUDA(cudaMalloc(&pBuf, pBufSize));

    BSRI_CHECK_CUSPARSE(cusparseDbsrilu02_analysis(
        spH, CUSPARSE_DIRECTION_ROW, nb, nnzb,
        descrA, d_bsrVal, d_bsrRow, d_bsrCol, BS,
        infoM, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuf));

    int zeroPivot = -1;
    auto zp = cusparseXbsrilu02_zeroPivot(spH, infoM, &zeroPivot);
    if (zp == CUSPARSE_STATUS_ZERO_PIVOT) {
        std::cerr << "[bsrilu02] WARNING structural zero at block ("
                  << zeroPivot << "," << zeroPivot << ")\n";
    }

    BSRI_CHECK_CUSPARSE(cusparseDbsrilu02(
        spH, CUSPARSE_DIRECTION_ROW, nb, nnzb,
        descrA, d_bsrVal, d_bsrRow, d_bsrCol, BS,
        infoM, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuf));

    zp = cusparseXbsrilu02_zeroPivot(spH, infoM, &zeroPivot);
    if (zp == CUSPARSE_STATUS_ZERO_PIVOT) {
        std::cerr << "[bsrilu02] ERROR numerical zero at U("
                  << zeroPivot << "," << zeroPivot << ")\n";
    }

    cudaFree(pBuf);
    cusparseDestroyBsrilu02Info(infoM);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(spH);
}

#else
// ======================== float version ========================
inline void bsrilu02_inplace(
    int nb, int nnzb, int BS,
    const int* d_bsrRow, const int* d_bsrCol,
    float* d_bsrVal)
{
    cusparseHandle_t spH; BSRI_CHECK_CUSPARSE(cusparseCreate(&spH));
    cusparseMatDescr_t descrA; BSRI_CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    BSRI_CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    BSRI_CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));

    bsrilu02Info_t infoM = 0; BSRI_CHECK_CUSPARSE(cusparseCreateBsrilu02Info(&infoM));

    int pBufSize = 0;
    BSRI_CHECK_CUSPARSE(cusparseSbsrilu02_bufferSize(
        spH, CUSPARSE_DIRECTION_ROW, nb, nnzb,
        descrA, d_bsrVal, d_bsrRow, d_bsrCol, BS, infoM, &pBufSize));

    void* pBuf = nullptr; BSRI_CHECK_CUDA(cudaMalloc(&pBuf, pBufSize));

    BSRI_CHECK_CUSPARSE(cusparseSbsrilu02_analysis(
        spH, CUSPARSE_DIRECTION_ROW, nb, nnzb,
        descrA, d_bsrVal, d_bsrRow, d_bsrCol, BS,
        infoM, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuf));

    int zeroPivot = -1;
    auto zp = cusparseXbsrilu02_zeroPivot(spH, infoM, &zeroPivot);
    if (zp == CUSPARSE_STATUS_ZERO_PIVOT) {
        std::cerr << "[bsrilu02] WARNING structural zero at block ("
                  << zeroPivot << "," << zeroPivot << ")\n";
    }

    BSRI_CHECK_CUSPARSE(cusparseSbsrilu02(
        spH, CUSPARSE_DIRECTION_ROW, nb, nnzb,
        descrA, d_bsrVal, d_bsrRow, d_bsrCol, BS,
        infoM, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuf));

    zp = cusparseXbsrilu02_zeroPivot(spH, infoM, &zeroPivot);
    if (zp == CUSPARSE_STATUS_ZERO_PIVOT) {
        std::cerr << "[bsrilu02] ERROR numerical zero at U("
                  << zeroPivot << "," << zeroPivot << ")\n";
    }

    cudaFree(pBuf);
    cusparseDestroyBsrilu02Info(infoM);
    cusparseDestroyMatDescr(descrA);
    cusparseDestroy(spH);
}
#endif
