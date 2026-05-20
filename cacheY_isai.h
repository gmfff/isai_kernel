#include "common.h"
#include <cstdio>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <cusparse_v2.h>   
#include <cublas_v2.h> 
#include "config.h"
#include <cuda_fp16.h>
#include <mma.h>


__device__ __forceinline__
void mma_m8n8k4_f64_new(double *acc, double frag_a, double frag_b)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 "
        "{ %0, %1 }, { %2 }, { %3 }, { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1])
        : "d"(frag_a), "d"(frag_b)
    );
#endif
}

__device__ __forceinline__
double load_fragA_from_sA_5x5_first4cols_new(const double* sA, int lane)
{
    int row = lane >> 2;   // 0..7
    int col = lane &  3;   // 0..3

    if (row < 5) {
        return sA[row * 5 + col];
    } else {
        return 0.0;
    }
}

// B_embed: 4x8
// 从 Yp(5x5,row-major) 里取前4行，组成 4x5；再补到 4x8
__device__ __forceinline__
double load_fragB_from_Yp_5x5_first4rows_new(const double* Yp, int lane)
{
    int row = lane &  3;   // 0..3
    int col = lane >> 2;   // 0..7

    if (col < 5) {
        return Yp[row * 5 + col];
    } else {
        return 0.0;
    }
}

// 只把左上 5x5 映射到 sRhs，其余补0
__device__ __forceinline__
void load_acc_from_sRhs_5x5_new(const double* sRhs, int lane, double acc[2])
{
    int groupID           = lane >> 2;  // 0..7 -> row
    int threadID_in_group = lane &  3;  // 0..3

    int row = groupID;
    int c0  = threadID_in_group * 2 + 0;
    int c1  = threadID_in_group * 2 + 1;

    acc[0] = (row < 5 && c0 < 5) ? sRhs[row * 5 + c0] : 0.0;
    acc[1] = (row < 5 && c1 < 5) ? sRhs[row * 5 + c1] : 0.0;
}

// 只把左上 5x5 写回 sRhs
__device__ __forceinline__
void store_acc_to_sRhs_5x5_new(double* sRhs, int lane, const double acc[2])
{
    int groupID           = lane >> 2;
    int threadID_in_group = lane &  3;

    int row = groupID;
    int c0  = threadID_in_group * 2 + 0;
    int c1  = threadID_in_group * 2 + 1;

    if (row < 5 && c0 < 5) sRhs[row * 5 + c0] = acc[0];
    if (row < 5 && c1 < 5) sRhs[row * 5 + c1] = acc[1];
}

// 做：sRhs -= A_use(5x4) * Yp_use(4x5)
// A 和 Yp 都从 shared/给定指针读
__device__ __forceinline__
void tc_update_rhs_5x5_k4_new(double* sRhs, const double* sA, const double* Yp)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    int lane = threadIdx.x & 31;

    double fragA = load_fragA_from_sA_5x5_first4cols_new(sA, lane);
    double fragB = load_fragB_from_Yp_5x5_first4rows(Yp, lane);

    double acc[2];
    load_acc_from_sRhs_5x5_new(sRhs, lane, acc);

    fragA = -fragA;
    mma_m8n8k4_f64_new(acc, fragA, fragB);

    store_acc_to_sRhs_5x5_new(sRhs, lane, acc);
#endif
}

__device__ __forceinline__
void add_tail_k4_outer_product_new(double* sRhs, const double* sA, const double* Yp)
{
    int lane = threadIdx.x & 31;

    if (lane < 25) {
        int rr = lane / 5;
        int cc = lane % 5;
        sRhs[rr * 5 + cc] -= sA[rr * 5 + 4] * Yp[4 * 5 + cc];
    }
}

// fallback 路径：sA shared, sYp 直接 global 普通读取
__device__ __forceinline__
void tc_update_rhs_5x5_k4_sA_shm_Yp_reg(double* sRhs,
                                        const double* sA,
                                        const double* Yp)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    int lane = threadIdx.x & 31;

    double fragA = load_fragA_from_sA_5x5_first4cols_new(sA, lane);

    int row = lane & 3;
    int col = lane >> 2;
    double fragB = (col < 5) ? Yp[row * 5 + col] : 0.0;

    double acc[2];
    load_acc_from_sRhs_5x5_new(sRhs, lane, acc);

    fragA = -fragA;
    mma_m8n8k4_f64_new(acc, fragA, fragB);

    store_acc_to_sRhs_5x5_new(sRhs, lane, acc);
#endif
}

__device__ __forceinline__
void add_tail_k4_outer_product_sA_shm_Yp_reg(double* sRhs,
                                             const double* sA,
                                             const double* Yp)
{
    int lane = threadIdx.x & 31;

    if (lane < 25) {
        int rr = lane / 5;
        int cc = lane % 5;
        double a = sA[rr * 5 + 4];
        double b = Yp[4 * 5 + cc];
        sRhs[rr * 5 + cc] -= a * b;
    }
}

__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY_tensor_hybrid(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    const int* __restrict__ Mbrow,
    const int* __restrict__ Mcol,
    const int* __restrict__ PairPtr,
    const int* __restrict__ PairPLocal,
    const int* __restrict__ PairSrc,
    VALUE_TYPE* __restrict__ Mval,
    const VALUE_TYPE* __restrict__ DinvL,
    int Nmax_perwarp
){
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;

    int j = blockIdx.x * warpsPerBlock + warp;
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N <= 0) return;

    extern __shared__ VALUE_TYPE sh[];
    VALUE_TYPE* warp_base = sh + warp * ((2 * BS2) + Nmax_perwarp * BS2);

    VALUE_TYPE* sRhs  = warp_base + 0;
    VALUE_TYPE* sA    = warp_base + BS2;
    VALUE_TYPE* sYall = warp_base + 2 * BS2;

    const bool use_cachedY = (N <= Nmax_perwarp);

    if (use_cachedY) {
        // ============================================================
        // 路径 A：full cachedY
        // ============================================================
        for (int idx = 0; idx < N; ++idx) {
            int t = Mcol[start + idx];

            if (lane < BS2) {
                int r = lane / BS;
                int c = lane - r * BS;
                sRhs[lane] = (t == j && r == c) ? (VALUE_TYPE)1.0 : (VALUE_TYPE)0.0;
            }
            __syncwarp();

            int g0   = start + idx;
            int qbeg = PairPtr[g0];
            int qend = PairPtr[g0 + 1];

            for (int q = qbeg; q < qend; ++q) {
                int p_local = PairPLocal[q];
                int srcA    = PairSrc[q];

                if (lane < BS2) {
                    sA[lane] = Abval[srcA * BS2 + lane];
                }
                __syncwarp();

                const VALUE_TYPE* Yp = sYall + p_local * BS2;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                tc_update_rhs_5x5_k4_new((double*)sRhs,
                                     (const double*)sA,
                                     (const double*)Yp);
                __syncwarp();

                add_tail_k4_outer_product_new((double*)sRhs,
                                          (const double*)sA,
                                          (const double*)Yp);
                __syncwarp();
#else
                if (lane < BS2) {
                    int rr = lane / BS;
                    int cc = lane - rr * BS;
                    VALUE_TYPE sum = (VALUE_TYPE)0.0;
#pragma unroll
                    for (int k = 0; k < BS; ++k) {
                        sum += sA[rr * BS + k] * Yp[k * BS + cc];
                    }
                    sRhs[lane] -= sum;
                }
                __syncwarp();
#endif
            }

            if (lane < BS2) {
                const VALUE_TYPE* inv = DinvL + t * BS2;
                int rr = lane / BS;
                int cc = lane - rr * BS;
                VALUE_TYPE acc = (VALUE_TYPE)0.0;

#pragma unroll
                for (int k = 0; k < BS; ++k) {
                    acc += inv[rr * BS + k] * sRhs[k * BS + cc];
                }

                sYall[idx * BS2 + lane] = acc;
            }
            __syncwarp();
        }

        for (int idx = 0; idx < N; ++idx) {
            if (lane < BS2) {
                Mval[(start + idx) * BS2 + lane] = sYall[idx * BS2 + lane];
            }
            __syncwarp();
        }

    } else {
        // ============================================================
        // 路径 B：fallback，sA shared, Yp global
        // ============================================================
        for (int idx = 0; idx < N; ++idx) {
            int t = Mcol[start + idx];

            if (lane < BS2) {
                int r = lane / BS;
                int c = lane - r * BS;
                sRhs[lane] = (t == j && r == c) ? (VALUE_TYPE)1.0 : (VALUE_TYPE)0.0;
            }
            __syncwarp();

            int g0   = start + idx;
            int qbeg = PairPtr[g0];
            int qend = PairPtr[g0 + 1];

            for (int q = qbeg; q < qend; ++q) {
                int p_local = PairPLocal[q];
                int srcA    = PairSrc[q];

                if (lane < BS2) {
                    sA[lane] = Abval[srcA * BS2 + lane];
                }
                __syncwarp();

                const VALUE_TYPE* Yp = Mval + (start + p_local) * BS2;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
                tc_update_rhs_5x5_k4_sA_shm_Yp_reg((double*)sRhs,
                                                   (const double*)sA,
                                                   (const double*)Yp);
                __syncwarp();

                add_tail_k4_outer_product_sA_shm_Yp_reg((double*)sRhs,
                                                        (const double*)sA,
                                                        (const double*)Yp);
                __syncwarp();
#else
                if (lane < BS2) {
                    int rr = lane / BS;
                    int cc = lane - rr * BS;
                    VALUE_TYPE sum = (VALUE_TYPE)0.0;
#pragma unroll
                    for (int k = 0; k < BS; ++k) {
                        sum += sA[rr * BS + k] * Yp[k * BS + cc];
                    }
                    sRhs[lane] -= sum;
                }
                __syncwarp();
#endif
            }

            if (lane < BS2) {
                const VALUE_TYPE* inv = DinvL + t * BS2;
                int rr = lane / BS;
                int cc = lane - rr * BS;
                VALUE_TYPE acc = (VALUE_TYPE)0.0;

#pragma unroll
                for (int k = 0; k < BS; ++k) {
                    acc += inv[rr * BS + k] * sRhs[k * BS + cc];
                }

                Mval[(start + idx) * BS2 + lane] = acc;
            }
            __syncwarp();
        }
    }
}
