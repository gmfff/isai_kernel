#include "common.h"
#include "csr2bsr.h"
#include "read_matrix.h"
#include "isai_bsr.h"
#include "cacheY_isai.h"
#include "bsrilu.h"
#include "bsrilu_debug.h"
#include "spmv_debug.h"
#include "residual_check.h"
#include "BILU_cusparse.h"
#include "bcsc2bcsrAndSpMV.h"
#include "bcsc2bcsr.h"
#include "bsr_tensor_SpMV.h"
#include <cusparse_v2.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <algorithm>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include "analyze_pair.h"

struct IsaiKernelTimes {
    float lower = 0.0f;
    float lower_tensor = 0.0f;
    float lower_globalY_tensor = 0.0f;
    float upper = 0.0f;
};

#ifndef CHECK_CUSPARSE
#define CHECK_CUSPARSE(call) do { \
    auto st = (call); \
    if (st != CUSPARSE_STATUS_SUCCESS) { \
        std::cerr << "cuSPARSE " << int(st) << " @ " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while (0)
#endif

#ifndef CHECK_CUBLAS
#define CHECK_CUBLAS(call) do { \
    auto st = (call); \
    if (st != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS " << int(st) << " @ " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while (0)
#endif

static void run_isai_kernels_bs4(
    int nb,
    int Nmax_perwarp,
    const int* d_Abrow,
    const int* d_Abcol,
    const VALUE_TYPE* d_Abval,
    const int* d_MbrowL,
    const int* d_McolL,
    VALUE_TYPE* d_MvalL,
    const int* d_MbrowU,
    const int* d_McolU,
    VALUE_TYPE* d_MvalU,
    const int* d_PairPtr,
    const int* d_PairPLocal,
    const int* d_PairSrc,
    const VALUE_TYPE* d_DinvL,
    VALUE_TYPE*& d_DinvU,
    IsaiKernelTimes& times,
    cudaEvent_t start1,
    cudaEvent_t stop1,
    cudaEvent_t start2,
    cudaEvent_t stop2,
    cudaEvent_t start5,
    cudaEvent_t stop5,
    cudaEvent_t start6,
    cudaEvent_t stop6)
{
    int warpsPerBlock = 1;
    dim3 block(32 * warpsPerBlock);
    dim3 grid((nb + warpsPerBlock - 1) / warpsPerBlock);

    size_t shmem_bytes =
        (size_t)warpsPerBlock * ((2 * BS2) + Nmax_perwarp * BS2) * sizeof(VALUE_TYPE);
    size_t shmem_tensor_bytes = shmem_bytes;

    int warpsPerBlockGlobalY = 4;
    dim3 blockGlobalY(32 * warpsPerBlockGlobalY);
    dim3 gridGlobalY((nb + warpsPerBlockGlobalY - 1) / warpsPerBlockGlobalY);
    size_t shmemGlobalY =
        (size_t)warpsPerBlockGlobalY * (2 * BS2) * sizeof(VALUE_TYPE);

    for (int i = 0; i < BENCH_WARMUP_ITERS; i++) {
        isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY<<<grid, block, shmem_bytes>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
            d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL,
            Nmax_perwarp);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start1, 0));
    for (int i = 0; i < BENCH_TIMING_ITERS; i++) {
        isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY<<<grid, block, shmem_bytes>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
            d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL,
            Nmax_perwarp);
    }
    CHECK_CUDA(cudaEventRecord(stop1, 0));
    CHECK_CUDA(cudaEventSynchronize(stop1));
    CHECK_CUDA(cudaEventElapsedTime(&times.lower, start1, stop1));
    CHECK_CUDA(cudaGetLastError());

    for (int i = 0; i < BENCH_WARMUP_ITERS; i++) {
        isai_lower_bsr4_warpcol_with_pairs_kernel_cachedY_tensor_notail<<<grid, block, shmem_tensor_bytes>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
            d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL,
            Nmax_perwarp);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start5, 0));
    for (int i = 0; i < BENCH_TIMING_ITERS; i++) {
        isai_lower_bsr4_warpcol_with_pairs_kernel_cachedY_tensor_notail<<<grid, block, shmem_tensor_bytes>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
            d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL,
            Nmax_perwarp);
    }
    CHECK_CUDA(cudaEventRecord(stop5, 0));
    CHECK_CUDA(cudaEventSynchronize(stop5));
    CHECK_CUDA(cudaEventElapsedTime(&times.lower_tensor, start5, stop5));
    CHECK_CUDA(cudaGetLastError());

    for (int i = 0; i < BENCH_WARMUP_ITERS; i++) {
        isai_lower_bsr4_warpcol_with_pairs_kernel_globalY_tensor_notail
            <<<gridGlobalY, blockGlobalY, shmemGlobalY>>>(
                nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
                d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start6, 0));
    for (int i = 0; i < BENCH_TIMING_ITERS; i++) {
        isai_lower_bsr4_warpcol_with_pairs_kernel_globalY_tensor_notail
            <<<gridGlobalY, blockGlobalY, shmemGlobalY>>>(
                nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
                d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL);
    }
    CHECK_CUDA(cudaEventRecord(stop6, 0));
    CHECK_CUDA(cudaEventSynchronize(stop6));
    CHECK_CUDA(cudaEventElapsedTime(&times.lower_globalY_tensor, start6, stop6));
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMalloc(&d_DinvU, nb * BS2 * sizeof(VALUE_TYPE)));
    {
        dim3 g((nb + 127) / 128), b(128);
        build_DinvU_from_BSR5_kernel_fix<<<g, b>>>(nb, d_Abrow, d_Abcol, d_Abval, d_DinvU);
        CHECK_CUDA(cudaGetLastError());
    }

    int warpsPerBlockU = 4;
    dim3 blockU(32 * warpsPerBlockU);
    dim3 gridU((nb + warpsPerBlockU - 1) / warpsPerBlockU);
    size_t shmemU = (size_t)warpsPerBlockU * (3 * BS2) * sizeof(VALUE_TYPE);

    for (int i = 0; i < BENCH_WARMUP_ITERS; i++) {
        isai_upper_bsr5_warpcol_with_Dinv_kernel<<<gridU, blockU, shmemU>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowU, d_McolU, d_MvalU, d_DinvU);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start2, 0));
    for (int i = 0; i < BENCH_TIMING_ITERS; i++) {
        isai_upper_bsr5_warpcol_with_Dinv_kernel<<<gridU, blockU, shmemU>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowU, d_McolU, d_MvalU, d_DinvU);
    }
    CHECK_CUDA(cudaEventRecord(stop2, 0));
    CHECK_CUDA(cudaEventSynchronize(stop2));
    CHECK_CUDA(cudaEventElapsedTime(&times.upper, start2, stop2));
    CHECK_CUDA(cudaGetLastError());
}

static void run_isai_kernels_bs5(
    int nb,
    int Nmax_perwarp,
    const int* d_Abrow,
    const int* d_Abcol,
    const VALUE_TYPE* d_Abval,
    const int* d_MbrowL,
    const int* d_McolL,
    VALUE_TYPE* d_MvalL,
    const int* d_MbrowU,
    const int* d_McolU,
    VALUE_TYPE* d_MvalU,
    const int* d_PairPtr,
    const int* d_PairPLocal,
    const int* d_PairSrc,
    const VALUE_TYPE* d_DinvL,
    VALUE_TYPE*& d_DinvU,
    IsaiKernelTimes& times,
    cudaEvent_t start1,
    cudaEvent_t stop1,
    cudaEvent_t start2,
    cudaEvent_t stop2,
    cudaEvent_t start5,
    cudaEvent_t stop5,
    cudaEvent_t start6,
    cudaEvent_t stop6)
{
    int warpsPerBlock = 1;
    dim3 block(32 * warpsPerBlock);
    dim3 grid((nb + warpsPerBlock - 1) / warpsPerBlock);

    size_t shmem_bytes =
        (size_t)warpsPerBlock * ((2 * BS2) + Nmax_perwarp * BS2) * sizeof(VALUE_TYPE);
    size_t shmem_tensor_bytes = shmem_bytes;

    int warpsPerBlockGlobalY = 4;
    dim3 blockGlobalY(32 * warpsPerBlockGlobalY);
    dim3 gridGlobalY((nb + warpsPerBlockGlobalY - 1) / warpsPerBlockGlobalY);
    size_t shmemGlobalY =
        (size_t)warpsPerBlockGlobalY * (2 * BS2) * sizeof(VALUE_TYPE);

    for (int i = 0; i < BENCH_WARMUP_ITERS; i++) {
        isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY<<<grid, block, shmem_bytes>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
            d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL,
            Nmax_perwarp);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start1, 0));
    for (int i = 0; i < BENCH_TIMING_ITERS; i++) {
        isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY<<<grid, block, shmem_bytes>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
            d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL,
            Nmax_perwarp);
    }
    CHECK_CUDA(cudaEventRecord(stop1, 0));
    CHECK_CUDA(cudaEventSynchronize(stop1));
    CHECK_CUDA(cudaEventElapsedTime(&times.lower, start1, stop1));
    CHECK_CUDA(cudaGetLastError());

    for (int i = 0; i < BENCH_WARMUP_ITERS; i++) {
        isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY_tensor<<<grid, block, shmem_tensor_bytes>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
            d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL,
            Nmax_perwarp);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start5, 0));
    for (int i = 0; i < BENCH_TIMING_ITERS; i++) {
        isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY_tensor<<<grid, block, shmem_tensor_bytes>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
            d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL,
            Nmax_perwarp);
    }
    CHECK_CUDA(cudaEventRecord(stop5, 0));
    CHECK_CUDA(cudaEventSynchronize(stop5));
    CHECK_CUDA(cudaEventElapsedTime(&times.lower_tensor, start5, stop5));
    CHECK_CUDA(cudaGetLastError());

    for (int i = 0; i < BENCH_WARMUP_ITERS; i++) {
        isai_lower_bsr5_warpcol_with_pairs_kernel_globalY_tensor
            <<<gridGlobalY, blockGlobalY, shmemGlobalY>>>(
                nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
                d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start6, 0));
    for (int i = 0; i < BENCH_TIMING_ITERS; i++) {
        isai_lower_bsr5_warpcol_with_pairs_kernel_globalY_tensor
            <<<gridGlobalY, blockGlobalY, shmemGlobalY>>>(
                nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
                d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL);
    }
    CHECK_CUDA(cudaEventRecord(stop6, 0));
    CHECK_CUDA(cudaEventSynchronize(stop6));
    CHECK_CUDA(cudaEventElapsedTime(&times.lower_globalY_tensor, start6, stop6));
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMalloc(&d_DinvU, nb * BS2 * sizeof(VALUE_TYPE)));
    {
        dim3 g((nb + 127) / 128), b(128);
        build_DinvU_from_BSR5_kernel_fix<<<g, b>>>(nb, d_Abrow, d_Abcol, d_Abval, d_DinvU);
        CHECK_CUDA(cudaGetLastError());
    }

    int warpsPerBlockU = 4;
    dim3 blockU(32 * warpsPerBlockU);
    dim3 gridU((nb + warpsPerBlockU - 1) / warpsPerBlockU);
    size_t shmemU = (size_t)warpsPerBlockU * (3 * BS2) * sizeof(VALUE_TYPE);

    for (int i = 0; i < BENCH_WARMUP_ITERS; i++) {
        isai_upper_bsr5_warpcol_with_Dinv_kernel<<<gridU, blockU, shmemU>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowU, d_McolU, d_MvalU, d_DinvU);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start2, 0));
    for (int i = 0; i < BENCH_TIMING_ITERS; i++) {
        isai_upper_bsr5_warpcol_with_Dinv_kernel<<<gridU, blockU, shmemU>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowU, d_McolU, d_MvalU, d_DinvU);
    }
    CHECK_CUDA(cudaEventRecord(stop2, 0));
    CHECK_CUDA(cudaEventSynchronize(stop2));
    CHECK_CUDA(cudaEventElapsedTime(&times.upper, start2, stop2));
    CHECK_CUDA(cudaGetLastError());
}

static void run_isai_kernels_bs7(
    int nb,
    int Nmax_perwarp,
    const int* d_Abrow,
    const int* d_Abcol,
    const VALUE_TYPE* d_Abval,
    const int* d_MbrowL,
    const int* d_McolL,
    VALUE_TYPE* d_MvalL,
    const int* d_MbrowU,
    const int* d_McolU,
    VALUE_TYPE* d_MvalU,
    const int* d_PairPtr,
    const int* d_PairPLocal,
    const int* d_PairSrc,
    const VALUE_TYPE* d_DinvL,
    VALUE_TYPE*& d_DinvU,
    IsaiKernelTimes& times,
    cudaEvent_t start1,
    cudaEvent_t stop1,
    cudaEvent_t start2,
    cudaEvent_t stop2,
    cudaEvent_t start5,
    cudaEvent_t stop5,
    cudaEvent_t start6,
    cudaEvent_t stop6)
{
    int warpsPerBlock = 1;
    dim3 block(32 * warpsPerBlock);
    dim3 grid((nb + warpsPerBlock - 1) / warpsPerBlock);

    size_t shmem_bytes =
        (size_t)warpsPerBlock * ((2 * BS2) + Nmax_perwarp * BS2) * sizeof(VALUE_TYPE);
    size_t shmem_tensor_bytes = shmem_bytes;

    int warpsPerBlockGlobalY = 4;
    dim3 blockGlobalY(32 * warpsPerBlockGlobalY);
    dim3 gridGlobalY((nb + warpsPerBlockGlobalY - 1) / warpsPerBlockGlobalY);
    size_t shmemGlobalY =
        (size_t)warpsPerBlockGlobalY * (2 * BS2) * sizeof(VALUE_TYPE);

    for (int i = 0; i < BENCH_WARMUP_ITERS; i++) {
        isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY<<<grid, block, shmem_bytes>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
            d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL,
            Nmax_perwarp);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start1, 0));
    for (int i = 0; i < BENCH_TIMING_ITERS; i++) {
        isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY<<<grid, block, shmem_bytes>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
            d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL,
            Nmax_perwarp);
    }
    CHECK_CUDA(cudaEventRecord(stop1, 0));
    CHECK_CUDA(cudaEventSynchronize(stop1));
    CHECK_CUDA(cudaEventElapsedTime(&times.lower, start1, stop1));
    CHECK_CUDA(cudaGetLastError());

    // BS=7 scheme 2:
    // first m8n8k4 handles k=0..3, second m8n8k4 handles k=4..6 with zero padding.
    for (int i = 0; i < BENCH_WARMUP_ITERS; i++) {
        isai_lower_bsr7_warpcol_with_pairs_kernel_cachedY_tensor_twomma<<<grid, block, shmem_tensor_bytes>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
            d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL,
            Nmax_perwarp);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start5, 0));
    for (int i = 0; i < BENCH_TIMING_ITERS; i++) {
        isai_lower_bsr7_warpcol_with_pairs_kernel_cachedY_tensor_twomma<<<grid, block, shmem_tensor_bytes>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
            d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL,
            Nmax_perwarp);
    }
    CHECK_CUDA(cudaEventRecord(stop5, 0));
    CHECK_CUDA(cudaEventSynchronize(stop5));
    CHECK_CUDA(cudaEventElapsedTime(&times.lower_tensor, start5, stop5));
    CHECK_CUDA(cudaGetLastError());

    for (int i = 0; i < BENCH_WARMUP_ITERS; i++) {
        isai_lower_bsr7_warpcol_with_pairs_kernel_globalY_tensor_twomma
            <<<gridGlobalY, blockGlobalY, shmemGlobalY>>>(
                nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
                d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL);
        CHECK_CUDA(cudaGetLastError());
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start6, 0));
    for (int i = 0; i < BENCH_TIMING_ITERS; i++) {
        isai_lower_bsr7_warpcol_with_pairs_kernel_globalY_tensor_twomma
            <<<gridGlobalY, blockGlobalY, shmemGlobalY>>>(
                nb, d_Abrow, d_Abcol, d_Abval, d_MbrowL, d_McolL,
                d_PairPtr, d_PairPLocal, d_PairSrc, d_MvalL, d_DinvL);
    }
    CHECK_CUDA(cudaEventRecord(stop6, 0));
    CHECK_CUDA(cudaEventSynchronize(stop6));
    CHECK_CUDA(cudaEventElapsedTime(&times.lower_globalY_tensor, start6, stop6));
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMalloc(&d_DinvU, nb * BS2 * sizeof(VALUE_TYPE)));
    {
        dim3 g((nb + 127) / 128), b(128);
        build_DinvU_from_BSR5_kernel_fix<<<g, b>>>(nb, d_Abrow, d_Abcol, d_Abval, d_DinvU);
        CHECK_CUDA(cudaGetLastError());
    }

    int warpsPerBlockU = 4;
    dim3 blockU(32 * warpsPerBlockU);
    dim3 gridU((nb + warpsPerBlockU - 1) / warpsPerBlockU);
    size_t shmemU = (size_t)warpsPerBlockU * (3 * BS2) * sizeof(VALUE_TYPE);

    for (int i = 0; i < BENCH_WARMUP_ITERS; i++) {
        isai_upper_bsr5_warpcol_with_Dinv_kernel<<<gridU, blockU, shmemU>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowU, d_McolU, d_MvalU, d_DinvU);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(start2, 0));
    for (int i = 0; i < BENCH_TIMING_ITERS; i++) {
        isai_upper_bsr5_warpcol_with_Dinv_kernel<<<gridU, blockU, shmemU>>>(
            nb, d_Abrow, d_Abcol, d_Abval, d_MbrowU, d_McolU, d_MvalU, d_DinvU);
    }
    CHECK_CUDA(cudaEventRecord(stop2, 0));
    CHECK_CUDA(cudaEventSynchronize(stop2));
    CHECK_CUDA(cudaEventElapsedTime(&times.upper, start2, stop2));
    CHECK_CUDA(cudaGetLastError());
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " matrix.mtx\n";
        return 0;
    }

    const std::string filename = argv[1];
    VALUE_TYPE val = VALUE_TYPE(1);
    std::cout << "VALUE_TYPE is: " << typeid(val).name() << std::endl;

    // =========================================================
    // 0. Event
    // =========================================================
    cudaEvent_t start1, stop1, start2, stop2, start3, stop3, start5, stop5, start6, stop6;
    float time_spmv_lower = 0.0f;

    CHECK_CUDA(cudaEventCreate(&start1));
    CHECK_CUDA(cudaEventCreate(&stop1));
    CHECK_CUDA(cudaEventCreate(&start2));
    CHECK_CUDA(cudaEventCreate(&stop2));
    CHECK_CUDA(cudaEventCreate(&start3));
    CHECK_CUDA(cudaEventCreate(&stop3));
    CHECK_CUDA(cudaEventCreate(&start5));
    CHECK_CUDA(cudaEventCreate(&stop5));
    CHECK_CUDA(cudaEventCreate(&start6));
    CHECK_CUDA(cudaEventCreate(&stop6));

    // =========================================================
    // 1. 读 mtx -> CSR
    // =========================================================
    int nrows = 0, ncols = 0;
    std::vector<int> csrRow, csrCol;
    std::vector<VALUE_TYPE> csrVal;

    if (!read_matrix_market_to_csr(filename, nrows, ncols, csrRow, csrCol, csrVal)) {
        return 0;
    }
    if (nrows != ncols) {
        std::cerr << "Matrix is not square, nrows=" << nrows
                  << ", ncols=" << ncols << "\n";
        return 0;
    }

    std::cout << "Read matrix " << filename
              << " of size " << nrows << " x " << ncols
              << ", nnz = " << csrVal.size() << "\n";

    // =========================================================
    // 2. CSR -> BSR(BS x BS)
    // =========================================================
    int nb = 0;
    std::vector<int> bsrRow, bsrCol;
    std::vector<VALUE_TYPE> bsrVal;

    csr_to_bsr5(nrows, csrRow, csrCol, csrVal, nb, bsrRow, bsrCol, bsrVal);

    std::cout << "Converted to BSR with block size " << BS
              << ", nb = " << nb
              << ", nnzb(blocks) = " << bsrCol.size() << "\n";

    // =========================================================
    // 3. 构造 ISAI(L/U) 的 BCSC pattern
    // =========================================================
    std::vector<int> MbrowL, McolL;
    std::vector<int> MbrowU, McolU;

    build_bcsc_pattern_from_bsr_scalarILU(
        nb, bsrRow.data(), bsrCol.data(), true, MbrowL, McolL);
    build_bcsc_pattern_from_bsr_scalarILU(
        nb, bsrRow.data(), bsrCol.data(), false, MbrowU, McolU);

    const int nnzbL = MbrowL.back();
    const int nnzbU = MbrowU.back();

    std::cout << "ISAI(L) pattern: nnzb(blocks) = " << nnzbL << "\n";
    std::cout << "ISAI(U) pattern: nnzb(blocks) = " << nnzbU << "\n";

    // =========================================================
    // 4. 从 A 的 BCSR 提取 L/U 的 BCSR pattern，并构造 map_bcsc2bcsr
    // =========================================================
    std::vector<int> MbcsrRowL, MbcsrColL, mapL_bcsc2bcsr;
    std::vector<int> MbcsrRowU, MbcsrColU, mapU_bcsc2bcsr;

    build_triangular_bcsr_pattern_from_A(
        nb, bsrRow.data(), bsrCol.data(), true, MbcsrRowL, MbcsrColL);
    build_triangular_bcsr_pattern_from_A(
        nb, bsrRow.data(), bsrCol.data(), false, MbcsrRowU, MbcsrColU);

    if ((int)MbcsrColL.size() != nnzbL) {
        std::cerr << "L pattern mismatch: BCSR nnzb = " << MbcsrColL.size()
                  << ", BCSC nnzb = " << nnzbL << "\n";
        return 1;
    }
    if ((int)MbcsrColU.size() != nnzbU) {
        std::cerr << "U pattern mismatch: BCSR nnzb = " << MbcsrColU.size()
                  << ", BCSC nnzb = " << nnzbU << "\n";
        return 1;
    }

    build_map_bcsc2bcsr_host(
        nb, MbrowL, McolL, MbcsrRowL, MbcsrColL, mapL_bcsc2bcsr);
    build_map_bcsc2bcsr_host(
        nb, MbrowU, McolU, MbcsrRowU, MbcsrColU, mapU_bcsc2bcsr);

    // =========================================================
    // 5. 拷贝 A 到 GPU
    // =========================================================
    int *d_Abrow = nullptr, *d_Abcol = nullptr;
    VALUE_TYPE *d_Abval = nullptr;
    VALUE_TYPE *d_Abval_Aorig = nullptr;

    CHECK_CUDA(cudaMalloc(&d_Abrow, (nb + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_Abcol, bsrCol.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_Abval, bsrVal.size() * sizeof(VALUE_TYPE)));

    CHECK_CUDA(cudaMemcpy(d_Abrow, bsrRow.data(),
                          (nb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Abcol, bsrCol.data(),
                          bsrCol.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Abval, bsrVal.data(),
                          bsrVal.size() * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&d_Abval_Aorig, bsrVal.size() * sizeof(VALUE_TYPE)));
    CHECK_CUDA(cudaMemcpy(d_Abval_Aorig, d_Abval,
                          bsrVal.size() * sizeof(VALUE_TYPE),
                          cudaMemcpyDeviceToDevice));

    // =========================================================
    // 6. BSR-ILU(0)
    // =========================================================
    bsrilu02_inplace(
        nb,
        static_cast<int>(bsrCol.size()),
        BS,
        d_Abrow, d_Abcol, d_Abval);

    // 可选：cuSPARSE SpTRSV 路线测试
    VALUE_TYPE *x_bsrcusp = nullptr;
    VALUE_TYPE *ILU_cusp = nullptr;
    BILU_cuSPARSE(d_Abrow, d_Abcol, d_Abval, nrows, nb,
                  (int)bsrCol.size(), BS, &x_bsrcusp, &ILU_cusp);

    // =========================================================
    // 7. 分配 M_L / M_U 的 BCSC 与 BCSR 存储
    // =========================================================
    // ---- L: BCSC ----
    int *d_MbrowL = nullptr, *d_McolL = nullptr;
    VALUE_TYPE *d_MvalL = nullptr;

    CHECK_CUDA(cudaMalloc(&d_MbrowL, (nb + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_McolL, nnzbL * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_MvalL, nnzbL * BS2 * sizeof(VALUE_TYPE)));

    CHECK_CUDA(cudaMemcpy(d_MbrowL, MbrowL.data(),
                          (nb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_McolL, McolL.data(),
                          nnzbL * sizeof(int), cudaMemcpyHostToDevice));

    // ---- U: BCSC ----
    int *d_MbrowU = nullptr, *d_McolU = nullptr;
    VALUE_TYPE *d_MvalU = nullptr;

    CHECK_CUDA(cudaMalloc(&d_MbrowU, (nb + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_McolU, nnzbU * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_MvalU, nnzbU * BS2 * sizeof(VALUE_TYPE)));

    CHECK_CUDA(cudaMemcpy(d_MbrowU, MbrowU.data(),
                          (nb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_McolU, McolU.data(),
                          nnzbU * sizeof(int), cudaMemcpyHostToDevice));

    // ---- L: BCSR + map ----
    int *d_MbcsrRowL = nullptr, *d_MbcsrColL = nullptr;
    int *d_mapL_bcsc2bcsr = nullptr;
    VALUE_TYPE *d_MbcsrValL = nullptr;

    CHECK_CUDA(cudaMalloc(&d_MbcsrRowL, (nb + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_MbcsrColL, nnzbL * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_MbcsrValL, nnzbL * BS2 * sizeof(VALUE_TYPE)));
    CHECK_CUDA(cudaMalloc(&d_mapL_bcsc2bcsr, nnzbL * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_MbcsrRowL, MbcsrRowL.data(),
                          (nb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_MbcsrColL, MbcsrColL.data(),
                          nnzbL * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mapL_bcsc2bcsr, mapL_bcsc2bcsr.data(),
                          nnzbL * sizeof(int), cudaMemcpyHostToDevice));

    // ---- U: BCSR + map ----
    int *d_MbcsrRowU = nullptr, *d_MbcsrColU = nullptr;
    int *d_mapU_bcsc2bcsr = nullptr;
    VALUE_TYPE *d_MbcsrValU = nullptr;

    CHECK_CUDA(cudaMalloc(&d_MbcsrRowU, (nb + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_MbcsrColU, nnzbU * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_MbcsrValU, nnzbU * BS2 * sizeof(VALUE_TYPE)));
    CHECK_CUDA(cudaMalloc(&d_mapU_bcsc2bcsr, nnzbU * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_MbcsrRowU, MbcsrRowU.data(),
                          (nb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_MbcsrColU, MbcsrColU.data(),
                          nnzbU * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mapU_bcsc2bcsr, mapU_bcsc2bcsr.data(),
                          nnzbU * sizeof(int), cudaMemcpyHostToDevice));

    // =========================================================
    // 8. ISAI(L): 准备 DinvL + pair list
    // =========================================================
    VALUE_TYPE* d_DinvL = nullptr;
    CHECK_CUDA(cudaMalloc(&d_DinvL, nb * BS2 * sizeof(VALUE_TYPE)));
    {
        dim3 g((nb + 127) / 128), b(128);
        build_DinvL_from_BSR5_kernel_fix<<<g, b>>>(nb, d_Abrow, d_Abcol, d_Abval, d_DinvL);
        CHECK_CUDA(cudaGetLastError());
    }

    int Mnnz = 0;
    CHECK_CUDA(cudaMemcpy(&Mnnz, d_MbrowL + nb, sizeof(int), cudaMemcpyDeviceToHost));

    int* d_PairPtr = nullptr;
    int* d_PairPLocal = nullptr;
    int* d_PairSrc = nullptr;

    CHECK_CUDA(cudaMalloc(&d_PairPtr, (size_t)(Mnnz + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_PairPtr, 0, (size_t)(Mnnz + 1) * sizeof(int)));

    int warpsPerBlock = 1;
    dim3 block(32 * warpsPerBlock);
    dim3 grid((nb + warpsPerBlock - 1) / warpsPerBlock);

    count_pairs_lower_kernel_v2<<<grid, block>>>(
        nb, d_Abrow, d_Abcol, d_MbrowL, d_McolL, d_PairPtr);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUDA(cudaMemset(d_PairPtr + Mnnz, 0, sizeof(int)));
    thrust::exclusive_scan(thrust::device, d_PairPtr, d_PairPtr + (Mnnz + 1), d_PairPtr);

    int total_pairs = 0;
    CHECK_CUDA(cudaMemcpy(&total_pairs, d_PairPtr + Mnnz, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaMalloc(&d_PairPLocal, (size_t)total_pairs * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_PairSrc,    (size_t)total_pairs * sizeof(int)));

    std::vector<int> h_MbrowL(nb + 1);
    CHECK_CUDA(cudaMemcpy(h_MbrowL.data(), d_MbrowL,
                          (nb + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> h_PairPtr(Mnnz + 1);
    CHECK_CUDA(cudaMemcpy(h_PairPtr.data(), d_PairPtr,
                        (size_t)(Mnnz + 1) * sizeof(int),
                        cudaMemcpyDeviceToHost));
    int Nmax = 0;
    for (int j = 0; j < nb; ++j) {
        int N = h_MbrowL[j + 1] - h_MbrowL[j];
        if (N > Nmax) Nmax = N;
    }
    std::cout << "Nmax = " << Nmax << "\n";

    /*analyze_pair_lengths(nb, h_MbrowL, h_PairPtr);
    analyze_column_q_distribution(nb, h_MbrowL, h_PairPtr, 30);
    analyze_column_q_bins(nb, h_MbrowL, h_PairPtr, 10);
    analyze_last_columns(
        nb,
        h_MbrowL,
        h_PairPtr,
        50   // 看最后50列
    );*/

    int max_shared_memory = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&max_shared_memory,
                                      cudaDevAttrMaxSharedMemoryPerBlock, 0));
    std::cout << "Max shared memory per block: "
              << max_shared_memory << " bytes\n";

    fill_pairs_lower_kernel_v2<<<grid, block>>>(
        nb, d_Abrow, d_Abcol, d_MbrowL, d_McolL,
        d_PairPtr, d_PairPLocal, d_PairSrc);
    CHECK_CUDA(cudaGetLastError());

    unsigned int h_cnt = 0;
    for (int g = 0; g < Mnnz; ++g) {
        if (h_PairPtr[g + 1] - h_PairPtr[g] > 4) {
            ++h_cnt;
        }
    }
    std::cout << "Mnnz: " << Mnnz
              << " [INFO] entries with pair > 4 : "
              << h_cnt << " / " << total_pairs << "\n";

    VALUE_TYPE* d_DinvU = nullptr;
    IsaiKernelTimes isai_times;

    switch (BS) {
    case 4:
        run_isai_kernels_bs4(
            nb, Nmax, d_Abrow, d_Abcol, d_Abval,
            d_MbrowL, d_McolL, d_MvalL,
            d_MbrowU, d_McolU, d_MvalU,
            d_PairPtr, d_PairPLocal, d_PairSrc,
            d_DinvL, d_DinvU, isai_times,
            start1, stop1, start2, stop2, start5, stop5, start6, stop6);
        break;
    case 5:
        run_isai_kernels_bs5(
            nb, Nmax, d_Abrow, d_Abcol, d_Abval,
            d_MbrowL, d_McolL, d_MvalL,
            d_MbrowU, d_McolU, d_MvalU,
            d_PairPtr, d_PairPLocal, d_PairSrc,
            d_DinvL, d_DinvU, isai_times,
            start1, stop1, start2, stop2, start5, stop5, start6, stop6);
        break;
    case 7:
        run_isai_kernels_bs7(
            nb, Nmax, d_Abrow, d_Abcol, d_Abval,
            d_MbrowL, d_McolL, d_MvalL,
            d_MbrowU, d_McolU, d_MvalU,
            d_PairPtr, d_PairPLocal, d_PairSrc,
            d_DinvL, d_DinvU, isai_times,
            start1, stop1, start2, stop2, start5, stop5, start6, stop6);
        break;
    default:
        std::cerr << "Unsupported BS = " << BS << ". Supported values are 4, 5 and 7.\n";
        return 1;
    }

    std::cout << "ISAI(L) kernel avg time: " << isai_times.lower / BENCH_TIMING_ITERS << " ms\n";
    std::cout << "ISAI(L) tensor avg time: " << isai_times.lower_tensor / BENCH_TIMING_ITERS << " ms\n";
    std::cout << "ISAI(L) tensor globalY avg time: "
              << isai_times.lower_globalY_tensor / BENCH_TIMING_ITERS << " ms\n";
    std::cout << "ISAI(U) kernel avg time: " << isai_times.upper / BENCH_TIMING_ITERS << " ms\n";

    // =========================================================
    // 12. BCSC value -> BCSR value
    // =========================================================
    reorder_bcsc_val_to_bcsr_bs5<VALUE_TYPE><<<nnzbL, 32>>>(
        nnzbL, d_mapL_bcsc2bcsr, d_MvalL, d_MbcsrValL);
    CHECK_CUDA(cudaGetLastError());

    reorder_bcsc_val_to_bcsr_bs5<VALUE_TYPE><<<nnzbU, 32>>>(
        nnzbU, d_mapU_bcsc2bcsr, d_MvalU, d_MbcsrValU);
    CHECK_CUDA(cudaGetLastError());

    // =========================================================
    // 13. 应用：x = M_U * (M_L * b)
    // =========================================================
    const int n = nrows;

    std::vector<VALUE_TYPE> x_true(n, VALUE_TYPE(0.7));
    std::vector<VALUE_TYPE> b_host_new(n, VALUE_TYPE(0));

    for (int i = 0; i < n; ++i) {
        for (int p = csrRow[i]; p < csrRow[i + 1]; ++p) {
            int j = csrCol[p];
            b_host_new[i] += csrVal[p] * x_true[j];
        }
    }

    VALUE_TYPE *d_b = nullptr, *d_y = nullptr, *d_x = nullptr;
    CHECK_CUDA(cudaMalloc(&d_b, n * sizeof(VALUE_TYPE)));
    CHECK_CUDA(cudaMalloc(&d_y, n * sizeof(VALUE_TYPE)));
    CHECK_CUDA(cudaMalloc(&d_x, n * sizeof(VALUE_TYPE)));

    CHECK_CUDA(cudaMemcpy(d_b, b_host_new.data(),
                          n * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_y, 0, n * sizeof(VALUE_TYPE)));
    CHECK_CUDA(cudaMemset(d_x, 0, n * sizeof(VALUE_TYPE)));



    // =================================================
    // ISAI apply: y = M_L b, x = M_U y
    // =================================================
    int threads_spmv = 128;
    int blocks_spmv = (nb + threads_spmv - 1) / threads_spmv;
    for(int i=0;i<BENCH_WARMUP_ITERS;i++){
        bcsr_spmv<<<blocks_spmv, threads_spmv>>>(
            nb, BS, BS,
            d_MbcsrRowL, d_MbcsrColL, d_MbcsrValL,
            d_b, d_y);
    }
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaEventRecord(start3, 0));
    for(int i=0;i<BENCH_TIMING_ITERS;i++){
        bcsr_spmv<<<blocks_spmv, threads_spmv>>>(
            nb, BS, BS,
            d_MbcsrRowL, d_MbcsrColL, d_MbcsrValL,
            d_b, d_y);
    }
    CHECK_CUDA(cudaEventRecord(stop3, 0));
    CHECK_CUDA(cudaEventSynchronize(stop3));
    CHECK_CUDA(cudaEventElapsedTime(&time_spmv_lower, start3, stop3));
    CHECK_CUDA(cudaGetLastError());

    std::cout << "ISAI(L) SpMV kernel time: " << time_spmv_lower / BENCH_TIMING_ITERS << " ms\n";

    bcsr_spmv<<<blocks_spmv, threads_spmv>>>(
        nb, BS, BS,
        d_MbcsrRowU, d_MbcsrColU, d_MbcsrValU,
        d_y, d_x);
    //bsr_spmv_bdim5_kernel<<<blocks_spmv, threads_spmv>>>(d_MbcsrRowU, d_MbcsrColU, d_MbcsrValU,nb,nnzbU,d_y, d_x );
    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();
    // =========================================================
    // 14. 误差检查
    // =========================================================
    std::vector<VALUE_TYPE> x_approx(n);
    CHECK_CUDA(cudaMemcpy(x_approx.data(), d_x,
                          n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

    compute_rel_errors_csr<VALUE_TYPE>(
        nb * BS,
        csrRow.data(),
        csrCol.data(),
        csrVal.data(),
        b_host_new.data(),
        x_true.data(),
        d_x,
        "M_U M_L apply"
    );

    // =========================================================
    // 15. 可选：拷回 BCSC 的 M 值看看
    // =========================================================
    std::vector<VALUE_TYPE> h_MvalL(nnzbL * BS2);
    std::vector<VALUE_TYPE> h_MvalU(nnzbU * BS2);

    CHECK_CUDA(cudaMemcpy(h_MvalL.data(), d_MvalL,
                          nnzbL * BS2 * sizeof(VALUE_TYPE),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_MvalU.data(), d_MvalU,
                          nnzbU * BS2 * sizeof(VALUE_TYPE),
                          cudaMemcpyDeviceToHost));

    // =========================================================
    // 16. 清理
    // =========================================================
    CHECK_CUDA(cudaFree(d_Abrow));
    CHECK_CUDA(cudaFree(d_Abcol));
    CHECK_CUDA(cudaFree(d_Abval));
    CHECK_CUDA(cudaFree(d_Abval_Aorig));

    CHECK_CUDA(cudaFree(d_MbrowL));
    CHECK_CUDA(cudaFree(d_McolL));
    CHECK_CUDA(cudaFree(d_MvalL));

    CHECK_CUDA(cudaFree(d_MbrowU));
    CHECK_CUDA(cudaFree(d_McolU));
    CHECK_CUDA(cudaFree(d_MvalU));

    CHECK_CUDA(cudaFree(d_MbcsrRowL));
    CHECK_CUDA(cudaFree(d_MbcsrColL));
    CHECK_CUDA(cudaFree(d_MbcsrValL));
    CHECK_CUDA(cudaFree(d_mapL_bcsc2bcsr));

    CHECK_CUDA(cudaFree(d_MbcsrRowU));
    CHECK_CUDA(cudaFree(d_MbcsrColU));
    CHECK_CUDA(cudaFree(d_MbcsrValU));
    CHECK_CUDA(cudaFree(d_mapU_bcsc2bcsr));

    CHECK_CUDA(cudaFree(d_DinvL));
    CHECK_CUDA(cudaFree(d_DinvU));

    CHECK_CUDA(cudaFree(d_PairPtr));
    CHECK_CUDA(cudaFree(d_PairPLocal));
    CHECK_CUDA(cudaFree(d_PairSrc));

    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(d_x));

    // 如果 BILU_cuSPARSE 内部是 cudaMalloc 出来的，这两个也应释放
    // CHECK_CUDA(cudaFree(x_bsrcusp));
    // CHECK_CUDA(cudaFree(ILU_cusp));

    CHECK_CUDA(cudaEventDestroy(start1));
    CHECK_CUDA(cudaEventDestroy(stop1));
    CHECK_CUDA(cudaEventDestroy(start2));
    CHECK_CUDA(cudaEventDestroy(stop2));
    CHECK_CUDA(cudaEventDestroy(start3));
    CHECK_CUDA(cudaEventDestroy(stop3));
    CHECK_CUDA(cudaEventDestroy(start5));
    CHECK_CUDA(cudaEventDestroy(stop5));
    CHECK_CUDA(cudaEventDestroy(start6));
    CHECK_CUDA(cudaEventDestroy(stop6));

    CHECK_CUDA(cudaDeviceReset());
    return 0;
}
