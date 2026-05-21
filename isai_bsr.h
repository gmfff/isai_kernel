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
using namespace nvcuda;
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            std::cerr << "CUDA error " << cudaGetErrorString(err)          \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";    \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)


// ==================== device helpers ====================
__device__ __forceinline__
void mma_m8n8k4_f64(double *acc, double frag_a, double frag_b)
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

__device__ __forceinline__ unsigned smem_u32addr(const void* p)
{
#if __CUDA_ARCH__ >= 800
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
#else
    return 0u;
#endif
}

template<int BYTES>
__device__ __forceinline__ void cp_async_cg(void* smem_dst, const void* gmem_src)
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], %2;\n"
        :
        : "r"(smem_u32addr(smem_dst)),
          "l"(gmem_src),
          "n"(BYTES)
    );
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

template<int N>
__device__ __forceinline__ void cp_async_wait_group()
{
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

// 25 doubles, each active lane copies one double (8B)
template<typename T>
__device__ __forceinline__ void prefetch_block25_A_cp_async(T* sDst, const T* gSrc, int lane)
{
#if __CUDA_ARCH__ >= 800
    static_assert(sizeof(T) == 8, "This implementation assumes 8-byte VALUE_TYPE");
    if (lane < 25) {
        cp_async_cg<8>(sDst + lane, gSrc + lane);
    }
#endif
}

__device__ __forceinline__
void set_I5(VALUE_TYPE* Y) {
    #pragma unroll
    for (int i = 0; i < BS2; ++i) Y[i] = 0.0;
    #pragma unroll
    for (int d = 0; d < BS; ++d) Y[d * BS + d] = 1.0;
}

__device__ __forceinline__
void gemm5(const VALUE_TYPE* A, const VALUE_TYPE* B, VALUE_TYPE* C) {
    #pragma unroll
    for (int i = 0; i < BS; ++i) {
        #pragma unroll
        for (int j = 0; j < BS; ++j) {
            VALUE_TYPE s = 0.0;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                s += A[i * BS + k] * B[k * BS + j];
            }
            C[i * BS + j] = s;
        }
    }
}

__device__ __forceinline__
void Y_minus_AB5(const VALUE_TYPE* A, const VALUE_TYPE* B, VALUE_TYPE* Y) {
    #pragma unroll
    for (int i = 0; i < BS; ++i) {
        #pragma unroll
        for (int j = 0; j < BS; ++j) {
            VALUE_TYPE s = 0.0;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                s += A[i * BS + k] * B[k * BS + j];
            }
            Y[i * BS + j] -= s;
        }
    }
}

// BSR lookup: full block copy，若不存在则全 0
__device__ __forceinline__
void bsr_lookup5_full(int br, int bc,
                      const int* __restrict__ brow,
                      const int* __restrict__ bcol,
                      const VALUE_TYPE* __restrict__ bval,
                      VALUE_TYPE* out) {
    int s = brow[br];
    int e = brow[br + 1] - 1;
    int pos = -1;
    while (s <= e) {
        int m = (s + e) >> 1;
        int c = bcol[m];
        if (c == bc) { pos = m; break; }
        if (c < bc) s = m + 1;
        else        e = m - 1;
    }
    if (pos >= 0) {
        const VALUE_TYPE* src = bval + pos * BS2;
        #pragma unroll
        for (int i = 0; i < BS2; ++i) out[i] = src[i];
    } else {
        #pragma unroll
        for (int i = 0; i < BS2; ++i) out[i] = 0.0;
    }
}

// 点式ILU + BSR 打包情形：L_kk 是"单位下三角块"，前代解 L * X = Y
__device__ __forceinline__
void forward_solve_unitL5(const VALUE_TYPE* L, VALUE_TYPE* Y) {
    #pragma unroll
    for (int col = 0; col < BS; ++col) {
        for (int i = 0; i < BS; ++i) {
            VALUE_TYPE s = Y[i * BS + col];
            #pragma unroll
            for (int k = 0; k < i; ++k) {
                s -= L[i * BS + k] * Y[k * BS + col];
            }
            // L_ii == 1
            Y[i * BS + col] = s;
        }
    }
}

// 上三角 U * X = Y
__device__ __forceinline__
void backward_solve_U5(const VALUE_TYPE* U, VALUE_TYPE* Y) {
    #pragma unroll
    for (int col = 0; col < BS; ++col) {
        for (int i = BS - 1; i >= 0; --i) {
            VALUE_TYPE s = Y[i * BS + col];
            #pragma unroll
            for (int k = i + 1; k < BS; ++k) {
                s -= U[i * BS + k] * Y[k * BS + col];
            }
            s /= U[i * BS + i];
            Y[i * BS + col] = s;
        }
    }
}

// ===== [STEP3] 5x5 稠密块求逆，用于 U 的对角块 =====
__device__ __forceinline__
bool invert5_dense(const VALUE_TYPE* A, VALUE_TYPE* Ainv, VALUE_TYPE eps=1e-14){
    VALUE_TYPE M[BS2], I[BS2];
    #pragma unroll
    for(int i=0;i<BS2;++i){ M[i]=A[i]; I[i]=0.0; }
    #pragma unroll
    for(int d=0; d<BS; ++d) I[d*BS+d]=1.0;

    for(int k=0;k<BS;++k){
        VALUE_TYPE piv = M[k*BS+k];
        if(fabs(piv)<eps) return false;      // 简单防护：必要时上层做微调
        VALUE_TYPE invp = 1.0/piv;
        #pragma unroll
        for(int j=0;j<BS;++j){ M[k*BS+j]*=invp; I[k*BS+j]*=invp; }
        #pragma unroll
        for(int i=0;i<BS;++i){
            if(i==k) continue;
            VALUE_TYPE f = M[i*BS+k];
            #pragma unroll
            for(int j=0;j<BS;++j){
                M[i*BS+j] -= f*M[k*BS+j];
                I[i*BS+j] -= f*I[k*BS+j];
            }
        }
    }
    #pragma unroll
    for(int i=0;i<BS2;++i) Ainv[i]=I[i];
    return true;
}


// ==================== ISAI(L) kernel ====================

// M_L approx L^{-1}，L 是下三角，存成 BSR(L)
__global__ void isai_lower_bsr5_scalarILU(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,   // BSR for L
    const int* __restrict__ Mbrow,      // BCSC pattern for M_L
    const int* __restrict__ Mcol,
    VALUE_TYPE* __restrict__ Mval           // output blocks of M_L
) {
    int j   = blockIdx.x;       // column block
    int tid = threadIdx.x;      // entry in this column
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N == 0 || tid >= N) return;

    int t = Mcol[start + tid];  // row block index

    VALUE_TYPE Y[BS2];
    #pragma unroll
    for (int i = 0; i < BS2; ++i) Y[i] = 0.0;
    if (tid == 0) set_I5(Y);    // e_j 的块形式

    __shared__ VALUE_TYPE sY[BS2];

    // forward substitution in block space
    for (int k = 0; k < N; ++k) {
        int ik = Mcol[start + k];

        // L_{t,k}
        VALUE_TYPE Ltk[BS2];
        bsr_lookup5_full(t, ik, Abrow, Abcol, Abval, Ltk);

        if (tid == k) {
            // diagonal block L_kk：单位下三角块（来源于点式ILU+打包）
            VALUE_TYPE Lkk[BS2];
            bsr_lookup5_full(ik, ik, Abrow, Abcol, Abval, Lkk);
            forward_solve_unitL5(Lkk, Y);  // Y = L_kk^{-1} * Y（隐式）
            #pragma unroll
            for (int i = 0; i < BS2; ++i) sY[i] = Y[i];
        }

        __syncthreads();

        if (tid > k) {
            Y_minus_AB5(Ltk, sY, Y);  // Y_t -= L_{t,k} * Y_k
        }

        __syncthreads();
    }

    VALUE_TYPE* dst = Mval + (start + tid) * BS2;
    #pragma unroll
    for (int i = 0; i < BS2; ++i) dst[i] = Y[i];
}

// ==================== ISAI(U) kernel ====================

// M_U approx U^{-1}，U 是上三角，存成 BSR(U)
// U 是上三角（来自 bsrilu02 的 RAW-BSR），M 是 BCSC(U)
// 修复点：RHS=I5 放在 diagPos 对应的线程，而不是 tid==0
__global__ void isai_upper_bsr5_scalarILU(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,   // BSR for U (RAW after bsrilu02)
    const int* __restrict__ Mbrow,      // BCSC pattern for M_U
    const int* __restrict__ Mcol,
    VALUE_TYPE* __restrict__ Mval           // output blocks of M_U
) {
    int j   = blockIdx.x;
    int tid = threadIdx.x;
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N == 0 || tid >= N) return;

    // ---- 关键修复：定位本列的对角位置（Mcol[start + diagPos] == j）----
    int diagPos = -1;
    #pragma unroll
    for (int k = 0; k < N; ++k) {
        if (Mcol[start + k] == j) { diagPos = k; break; }
    }
    if (diagPos < 0) {
        // 理论上不会发生（构模强制加了对角），加一道保险
        return;
    }

    int t = Mcol[start + tid];  // 本线程负责的行块

    // 每个线程维护自己的 Y（该列的一个 5x5 块列向量）
    VALUE_TYPE Y[BS2];
    #pragma unroll
    for (int i = 0; i < BS2; ++i) Y[i] = 0.0;

    // 只对"对角对应的线程"设置 RHS=I5，其它线程保持 0
    if (tid == diagPos) set_I5(Y);

    __shared__ VALUE_TYPE sY[BS2];

    // 逆序回代：从列内最后一个块开始（上三角）
    for (int k = N - 1; k >= 0; --k) {
        int ik = Mcol[start + k];

        // 读取 U_{t,k}
        VALUE_TYPE Utk[BS2];
        bsr_lookup5_full(t, ik, Abrow, Abcol, Abval, Utk);

        // 枢轴步：由 tid==k 的线程做 y_k ← U_kk^{-1} * y_k（这里用 backward_solve_U5）
        if (tid == k) {
            VALUE_TYPE Ukk[BS2];
            bsr_lookup5_full(ik, ik, Abrow, Abcol, Abval, Ukk);
            backward_solve_U5(Ukk, Y);  // Y = U_kk^{-1} * Y
            #pragma unroll
            for (int i = 0; i < BS2; ++i) sY[i] = Y[i];
        }

        __syncthreads();

        // 消元更新：t < k 的线程做 Y_t -= U_{t,k} * Y_k
        if (tid < k) {
            Y_minus_AB5(Utk, sY, Y);
        }

        __syncthreads();
    }

    // 写回本列该行块的结果
    VALUE_TYPE* dst = Mval + (start + tid) * BS2;
    #pragma unroll
    for (int i = 0; i < BS2; ++i) dst[i] = Y[i];
}


//  预计算 U 的对角块逆：DinvU[k] = (U_kk)^{-1} =====
__global__ void build_DinvU_from_BSR5_kernel(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    VALUE_TYPE* __restrict__ DinvU)   // nb × 25
{
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if(k>=nb) return;

    VALUE_TYPE Ukk[BS2], Uinv[BS2];
    bsr_lookup5_full(k, k, Abrow, Abcol, Abval, Ukk);
    bool ok = invert5_dense(Ukk, Uinv);
    if(!ok){
        // 兜底：非常小的对角加一点正则再逆
        #pragma unroll
        for(int d=0; d<BS; ++d) Ukk[d*BS+d] += 1e-12;
        invert5_dense(Ukk, Uinv);
    }
    VALUE_TYPE* dst = DinvU + k*BS2;
    #pragma unroll
    for(int i=0;i<BS2;++i) dst[i]=Uinv[i];
}

//  ISAI(U) - 使用 DinvU：用 GEMM 代替块内回代 =====
__global__ void isai_upper_bsr5_with_Dinv_kernel(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    const int* __restrict__ Mbrow,
    const int* __restrict__ Mcol,
    VALUE_TYPE* __restrict__ Mval,
    const VALUE_TYPE* __restrict__ DinvU)
{
    int j   = blockIdx.x;
    int tid = threadIdx.x;
    if (j >= nb) return;

    int start = Mbrow[j], end = Mbrow[j+1];
    int N = end - start;
    if (N == 0 || tid >= N) return;

    // ---- 关键修正：定位对角在本列的位置 ----
    __shared__ int sDiag;
    if (threadIdx.x == 0) {
        int pos = -1;
        for (int k = 0; k < N; ++k) {
            if (Mcol[start + k] == j) { pos = k; break; }
        }
        sDiag = pos; // 按我们的构模，一定 >=0；若意外<0，可直接 return
    }
    __syncthreads();
    if (sDiag < 0) return;

    int t = Mcol[start + tid];

    VALUE_TYPE Y[BS2];
    #pragma unroll
    for (int i = 0; i < BS2; ++i) Y[i] = 0.0;

    // 只给"对角对应的线程"放 RHS = I5
    if (tid == sDiag) set_I5(Y);

    __shared__ VALUE_TYPE sY[BS2];

    // 逆序回代（上三角）
    for (int k = N - 1; k >= 0; --k) {
        int ik = Mcol[start + k];

        VALUE_TYPE Utk[BS2];
        bsr_lookup5_full(t, ik, Abrow, Abcol, Abval, Utk);

         if (tid == k) {
            // Y_k ← U_kk^{-1} * Y_k
            VALUE_TYPE tmp[BS2];
            const VALUE_TYPE* inv = DinvU + ik*BS2;
            gemm5(inv, Y, tmp);
            #pragma unroll
            for (int i = 0; i < BS2; ++i) { Y[i] = tmp[i]; sY[i] = tmp[i]; }
        }
        __syncthreads();

        if (tid < k) {
            Y_minus_AB5(Utk, sY, Y); // Y_t -= U_tk * Y_k
        }
        __syncthreads();
    }

    VALUE_TYPE* dst = Mval + (start + tid) * BS2;
    #pragma unroll
    for (int i = 0; i < BS2; ++i) dst[i] = Y[i];
}
// U-ISAI（通用版）：一列内顺序回代，N 不受 blockDim.x 限制
__global__ void isai_upper_bsr5_with_Dinv_generic(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,   // BSR(LU)
    const int* __restrict__ Mbrow,      // BCSC(U)
    const int* __restrict__ Mcol,
    VALUE_TYPE* __restrict__ Mval,          // 输出（BCSC 顺序）
    const VALUE_TYPE* __restrict__ DinvU)   // nb×25
{
    int j = blockIdx.x;
    if (j >= nb) return;

    int start = Mbrow[j], end = Mbrow[j+1];
    int N = end - start;
    if (N <= 0) return;

    // 定位对角在本列的位置
    int diagPos = -1;
    for (int k = 0; k < N; ++k) if (Mcol[start + k] == j) { diagPos = k; break; }
    if (diagPos < 0) return; // 安全：理论上不应发生

    // 先把整列输出块清零
    for (int idx = threadIdx.x; idx < N; idx += blockDim.x) {
        VALUE_TYPE* dst = Mval + (start + idx) * BS2;
        #pragma unroll
        for (int i = 0; i < BS2; ++i) dst[i] = 0.0;
    }
    __syncthreads();

    // 共享临时
    __shared__ VALUE_TYPE sB[BS2];  // 本轮 RHS（5x5）
    __shared__ VALUE_TYPE sY[BS2];  // 本轮解 y_k（5x5）

    // 列回代：从后往前
    for (int k = N - 1; k >= 0; --k) {
        int ik = Mcol[start + k];

        if (threadIdx.x == 0) {
            // B = delta(k==diagPos)*I
            for (int i = 0; i < BS2; ++i) sB[i] = 0.0;
            if (k == diagPos) {
                for (int d = 0; d < BS; ++d) sB[d*BS + d] = 1.0;
            }
            // B -= sum_{m>k} U_{ik,im} * y_m
            for (int m = N - 1; m > k; --m) {
                int im = Mcol[start + m];
                VALUE_TYPE Ukm[BS2];
                bsr_lookup5_full(ik, im, Abrow, Abcol, Abval, Ukm);
                const VALUE_TYPE* Ym = Mval + (start + m) * BS2;
                // sB -= Ukm * Ym
                for (int r = 0; r < BS; ++r) {
                    for (int c = 0; c < BS; ++c) {
                        VALUE_TYPE s = 0.0;
                        #pragma unroll
                        for (int t = 0; t < BS; ++t) s += Ukm[r*BS + t] * Ym[t*BS + c];
                        sB[r*BS + c] -= s;
                    }
                }
            }
            // y_k = U_kk^{-1} * B
            const VALUE_TYPE* inv = DinvU + ik * BS2;
            for (int r = 0; r < BS; ++r) {
                for (int c = 0; c < BS; ++c) {
                    VALUE_TYPE s = 0.0;
                    #pragma unroll
                    for (int t = 0; t < BS; ++t) s += inv[r*BS + t] * sB[t*BS + c];
                    sY[r*BS + c] = s;
                }
            }
            // 写回本列第 k 个块
            VALUE_TYPE* dst = Mval + (start + k) * BS2;
            #pragma unroll
            for (int i = 0; i < BS2; ++i) dst[i] = sY[i];
        }
        __syncthreads();
    }
}


// ===== build DinvL from BSR5 (diagonal blocks inverse) =====
__global__ void build_DinvL_from_BSR5_kernel(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    VALUE_TYPE* __restrict__ DinvL)   // nb × 25
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nb) return;

    VALUE_TYPE Lkk[BS2], Linv[BS2];
    bsr_lookup5_full(k, k, Abrow, Abcol, Abval, Lkk);

    // 关键：unit lower block，强制对角为 1
    #pragma unroll
    for (int d = 0; d < BS; ++d) Lkk[d*BS + d] = 1.0;

    bool ok = invert5_dense(Lkk, Linv);
    if (!ok) {
        // 一般不会到这；兜底
        #pragma unroll
        for (int d = 0; d < BS; ++d) Lkk[d*BS + d] += 1e-12;
        invert5_dense(Lkk, Linv);
    }

    VALUE_TYPE* dst = DinvL + k * BS2;
    #pragma unroll
    for (int i = 0; i < BS2; ++i) dst[i] = Linv[i];
}



// ===== ISAI(L) using DinvL: forward substitution in block space =====
__global__ void isai_lower_bsr5_with_Dinv_kernel(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    const int* __restrict__ Mbrow,
    const int* __restrict__ Mcol,
    VALUE_TYPE* __restrict__ Mval,
    const VALUE_TYPE* __restrict__ DinvL)
{
    int j   = blockIdx.x;   // column block
    int tid = threadIdx.x;  // one row-block in this column
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N == 0 || tid >= N) return;

    // ---- locate diagonal position in this column pattern ----
    __shared__ int sDiag;
    if (threadIdx.x == 0) {
        int pos = -1;
        for (int k = 0; k < N; ++k) {
            if (Mcol[start + k] == j) { pos = k; break; }
        }
        sDiag = pos;
    }
    __syncthreads();
    if (sDiag < 0) return;

    int t = Mcol[start + tid]; // row-block index for this thread

    VALUE_TYPE Y[BS2];
    #pragma unroll
    for (int i = 0; i < BS2; ++i) Y[i] = 0.0;

    // RHS is I only for diagonal row-block (same as your upper)
    if (tid == sDiag) set_I5(Y);

    __shared__ VALUE_TYPE sY[BS2];

    // ---- forward substitution (lower): k = 0..N-1 ----
    for (int k = 0; k < N; ++k) {
        int ik = Mcol[start + k];

        // L_{t,ik}
        VALUE_TYPE Ltk[BS2];
        bsr_lookup5_full(t, ik, Abrow, Abcol, Abval, Ltk);

         if (tid == k) {
            // Y_k <- inv(L_kk) * Y_k
            VALUE_TYPE tmp[BS2];
            const VALUE_TYPE* inv = DinvL + ik * BS2;
            gemm5(inv, Y, tmp);
            #pragma unroll
            for (int i = 0; i < BS2; ++i) { Y[i] = tmp[i]; sY[i] = tmp[i]; }
        }

        __syncthreads();

        // for lower: rows below k get updated
        if (tid > k) {
            Y_minus_AB5(Ltk, sY, Y);  // Y_t -= L_{t,ik} * Y_k
        }
        __syncthreads();
    }

    VALUE_TYPE* dst = Mval + (start + tid) * BS2;
    #pragma unroll
    for (int i = 0; i < BS2; ++i) dst[i] = Y[i];
}

__global__ void build_DinvL_from_BSR5_kernel_fix(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    VALUE_TYPE* __restrict__ DinvL)   // nb×25
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nb) return;

    VALUE_TYPE Lkk[BS2], Linv[BS2];
    bsr_lookup5_full(k, k, Abrow, Abcol, Abval, Lkk);

    // 只保留下三角；上三角清零；对角=1
    #pragma unroll
    for (int i = 0; i < BS; ++i) {
        #pragma unroll
        for (int j = 0; j < BS; ++j) {
            if (j > i) Lkk[i*BS + j] = 0.0;   // upper -> 0
        }
        Lkk[i*BS + i] = 1.0;                 // unit diag
    }

    bool ok = invert5_dense(Lkk, Linv);
    if (!ok) {
        // 理论上不会：unit lower 必可逆；这里兜底
        #pragma unroll
        for (int d = 0; d < BS; ++d) Lkk[d*BS + d] += 1e-12;
        invert5_dense(Lkk, Linv);
    }

    VALUE_TYPE* dst = DinvL + k * BS2;
    #pragma unroll
    for (int i = 0; i < BS2; ++i) dst[i] = Linv[i];
}
__global__ void build_DinvU_from_BSR5_kernel_fix(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    VALUE_TYPE* __restrict__ DinvU)   // nb×25
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= nb) return;

    VALUE_TYPE Ukk[BS2], Uinv[BS2];
    bsr_lookup5_full(k, k, Abrow, Abcol, Abval, Ukk);

    // 只保留上三角；下三角清零
    #pragma unroll
    for (int i = 0; i < BS; ++i) {
        #pragma unroll
        for (int j = 0; j < BS; ++j) {
            if (j < i) Ukk[i*BS + j] = 0.0;   // lower -> 0
        }
    }

    bool ok = invert5_dense(Ukk, Uinv);
    if (!ok) {
        #pragma unroll
        for (int d = 0; d < BS; ++d) Ukk[d*BS + d] += 1e-12;
        invert5_dense(Ukk, Uinv);
    }

    VALUE_TYPE* dst = DinvU + k * BS2;
    #pragma unroll
    for (int i = 0; i < BS2; ++i) dst[i] = Uinv[i];
}


// 一个 warp 处理一列 j：顺序计算该列 pattern 里的每个块
// lane 0..24 负责 5x5 的 25 个元素并行做 GEMM/AXPY
__global__ void isai_lower_bsr5_warpcol_with_Dinv_kernel(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    const int* __restrict__ Mbrow,   // pattern for column j
    const int* __restrict__ Mcol,
    VALUE_TYPE* __restrict__ Mval,       // output blocks in BCSC order
    const VALUE_TYPE* __restrict__ DinvL // nb*25, 已按 unit-lower 语义构造的逆
)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;

    int j = blockIdx.x * warpsPerBlock + warp; // warp->column
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N <= 0) return;

    // shared per-warp: rhs(25) + Ltmp(25) + Yp(25)
    extern __shared__ VALUE_TYPE sh[];
    const int STRIDE = 3 * BS2; // 75 doubles per warp
    VALUE_TYPE* base = sh + warp * STRIDE;
    VALUE_TYPE* sRhs = base + 0 * BS2;
    VALUE_TYPE* sA   = base + 1 * BS2; // L(t, colp)
    VALUE_TYPE* sYp  = base + 2 * BS2; // previously computed Y_p

    // 顺序计算本列的每个块：idx = 0..N-1
    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];   // 当前要算的行块

        // rhs = (t==j ? I : 0)
        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? 1.0 : 0.0;
        }
        __syncwarp();

        // rhs -= sum_{p<idx} L(t, colp) * Y_p
        for (int p = 0; p < idx; ++p) {
            int colp = Mcol[start + p];

            // lane0: load A=L(t,colp) -> sA
            if (lane == 0) {
                bsr_lookup5_full(t, colp, Abrow, Abcol, Abval, sA);
            }

            // load Y_p from global -> sYp
            if (lane < BS2) {
                sYp[lane] = Mval[(start + p) * BS2 + lane];
            }
            __syncwarp();

            // sRhs -= sA * sYp  (每个 lane 算一个元素)
            if (lane < BS2) {
                int i = lane / BS;
                int j2 = lane - i * BS;
                VALUE_TYPE sum = 0.0;
                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    sum += sA[i * BS + k] * sYp[k * BS + j2];
                }
                sRhs[lane] -= sum;
            }
            __syncwarp();
        }

        // Y = DinvL[t] * rhs
        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            int i = lane / BS;
            int j2 = lane - i * BS;
            VALUE_TYPE sum = 0.0;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                sum += inv[i * BS + k] * sRhs[k * BS + j2];
            }
            // store result block
            Mval[(start + idx) * BS2 + lane] = sum;
        }
        __syncwarp();
    }
}

__global__ void isai_upper_bsr5_warpcol_with_Dinv_kernel(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    const int* __restrict__ Mbrow,
    const int* __restrict__ Mcol,
    VALUE_TYPE* __restrict__ Mval,
    const VALUE_TYPE* __restrict__ DinvU   // 已按 upper 语义(下三角清零)构造的逆
)
{
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
    const int STRIDE = 3 * BS2;
    VALUE_TYPE* base = sh + warp * STRIDE;
    VALUE_TYPE* sRhs = base + 0 * BS2;
    VALUE_TYPE* sA   = base + 1 * BS2; // U(t,colp)
    VALUE_TYPE* sYp  = base + 2 * BS2; // already computed y_p (p>idx)

    // 逆序：idx=N-1..0
    for (int idx = N - 1; idx >= 0; --idx) {
        int t = Mcol[start + idx];

        // rhs = (t==j ? I : 0)
        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? 1.0 : 0.0;
        }
        __syncwarp();

        // rhs -= sum_{p>idx} U(t, colp) * Y_p
        for (int p = N - 1; p > idx; --p) {
            int colp = Mcol[start + p];

            if (lane == 0) {
                bsr_lookup5_full(t, colp, Abrow, Abcol, Abval, sA);
            }
            if (lane < BS2) {
                sYp[lane] = Mval[(start + p) * BS2 + lane];
            }
            __syncwarp();

            if (lane < BS2) {
                int i = lane / BS;
                int j2 = lane - i * BS;
                VALUE_TYPE sum = 0.0;
                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    sum += sA[i * BS + k] * sYp[k * BS + j2];
                }
                sRhs[lane] -= sum;
            }
            __syncwarp();
        }

        // Y = DinvU[t] * rhs
        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvU + t * BS2;
            int i = lane / BS;
            int j2 = lane - i * BS;
            VALUE_TYPE sum = 0.0;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                sum += inv[i * BS + k] * sRhs[k * BS + j2];
            }
            Mval[(start + idx) * BS2 + lane] = sum;
        }
        __syncwarp();
    }
}
//tensor -core tensor-core tnesor-core
// 每个 warp：小块区 3*BS2（sRhs/sA/sYp） + WMMA 区 3*16*16（sA16/sB16/sC16）
// 数值类型：float（WMMA 用 TF32/FP32）
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_tc(
    int nb,
    const int*  __restrict__ Abrow,
    const int*  __restrict__ Abcol,
    const float*__restrict__ Abval,     // A 的块值 (float), BSR 顺序，每块 25 标量
    const int*  __restrict__ Mbrow,     // per-column rowptr
    const int*  __restrict__ Mcol,      // S_j
    const int*  __restrict__ PairPtr,   // 对列表 rowptr（与 Mbrow/Mcol 对齐）
    const int*  __restrict__ PairPLocal,// p 的列内局部号
    const int*  __restrict__ PairSrc,   // A 中 (t,p) 的 src_pos（Abcol 下标）
    float*      __restrict__ Mval,      // 输出（BCSC 顺序）
    const float*__restrict__ DinvL      // 预逆 (nb*25)
){
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    const int j = blockIdx.x * warpsPerBlock + warp;   // warp -> column
    if (j >= nb) return;

    const int start = Mbrow[j];
    const int end   = Mbrow[j+1];
    const int N     = end - start;
    if (N <= 0) return;

    // --- 每 warp 的共享内存布局：2*BS2(小缓冲) + [128B 对齐] + 3*16*16(WMMA 区) ---
    extern __shared__ unsigned char smem[];
    constexpr int BYTES_SMALL = 2 * BS2 * sizeof(float);               // 2*25*4 = 200B
    constexpr int ALIGN       = 128;
    constexpr int PAD         = (ALIGN - (BYTES_SMALL % ALIGN)) % ALIGN; // 56B
    constexpr int BYTES_WMMA  = 3 * 16 * 16 * sizeof(float);           // 3072B
    unsigned char* warp_base  = smem + warp * (BYTES_SMALL + PAD + BYTES_WMMA);

    // 小缓冲（5x5）用于把块装进共享（便于全 warp 协作写 16x16 的 TL 角落）
    float* sA   = reinterpret_cast<float*>(warp_base + 0 * BS2 * sizeof(float));
    float* sYp  = reinterpret_cast<float*>(warp_base + 1 * BS2 * sizeof(float));

    // 16x16 WMMA 缓冲（128B 对齐）
    float* sA16 = reinterpret_cast<float*>(warp_base + BYTES_SMALL + PAD + 0 * 16 * 16 * sizeof(float));
    float* sB16 = reinterpret_cast<float*>(warp_base + BYTES_SMALL + PAD + 1 * 16 * 16 * sizeof(float));
    float* sC16 = reinterpret_cast<float*>(warp_base + BYTES_SMALL + PAD + 2 * 16 * 16 * sizeof(float));

    // 每个 lane 对应的 (r,c)
    const int rr = lane / BS;
    const int cc = lane - rr * BS;

    // 活跃掩码（仅 0..24 参与）
    const unsigned FULL = 0xFFFFFFFFu;
    const unsigned act  = __ballot_sync(FULL, lane < BS2);

    // ---- 顺序处理本列的每个行块 ----
    for (int idx = 0; idx < N; ++idx) {
        const int t = Mcol[start + idx];

        // 该 lane 的 rhs_rc，初值为单位矩阵元素
        float rhs_rc = (lane < BS2 && t == j && rr == cc) ? 1.f : 0.f;

        // 该行的 (t,p) 列表
        const int g0   = start + idx;
        const int qbeg = PairPtr[g0];
        const int qend = PairPtr[g0 + 1];

        // 一次性清零 16x16 的 A/B（本 idx 内不再清；每次 pair 覆盖 TL 的 5x5）
        for (int t2 = lane; t2 < 16*16; t2 += 32) {
            sA16[t2] = 0.f;
            sB16[t2] = 0.f;
        }
        __syncwarp(act);

        // 本 lane 的 A*Yp 累积寄存器（C 的 TL(5x5) 元素累和）
        float acc_rc = 0.f;

        // ---- 遍历所有对：用 WMMA 做 C = A*Yp，并把 TL(5x5) 累加到 acc_rc ----
        for (int q = qbeg; q < qend; ++q) {
            const int p_local = PairPLocal[q];
            const int srcA    = PairSrc[q];

            // 载入 5x5 到小缓冲
            if (lane < BS2) {
                sA [lane] = Abval[srcA * BS2 + lane];
                sYp[lane] = Mval [(start + p_local) * BS2 + lane];
            }
            __syncwarp(act);

            // 覆盖写 TL 的 5x5 到 sA16/sB16（其余位置仍为 0）
            for (int e = lane; e < BS2; e += 32) {
                const int r = e / BS, c = e % BS;
                sA16[r*16 + c] = sA [r*BS + c];
                sB16[r*16 + c] = sYp[r*BS + c];
            }
            __syncwarp(act);

            // 一次 WMMA：C = A16 * B16；Cf 每次 fill 为 0，无需清 sC16
            wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::row_major> Af;
            wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::row_major> Bf;
            wmma::fragment<wmma::accumulator,       16,16,8, float> Cf;
            wmma::fill_fragment(Cf, 0.0f);
            wmma::load_matrix_sync(Af, sA16, 16);
            wmma::load_matrix_sync(Bf, sB16, 16);
            wmma::mma_sync(Cf, Af, Bf, Cf);
            wmma::store_matrix_sync(sC16, Cf, 16, wmma::mem_row_major);
            __syncwarp(act);

            // 累加 TL 的 5x5 到寄存器 acc_rc
            if (lane < BS2) {
                acc_rc += sC16[rr*16 + cc];
            }
            __syncwarp(act);
            // 注意：不清 sA16/sB16；下一轮 pair 继续覆盖 TL 5x5
        }

        // 一次性 rhs -= 累积 C
        if (lane < BS2) {
            rhs_rc -= acc_rc;
        }
        __syncwarp(act);

        // ---- 标量 5x5：Y = DinvL[t] * rhs（完全展开 + warp 内广播 rhs）----
        if (lane < BS2) {
            const float* inv = DinvL + t * BS2;   // 5x5
            float y = 0.f;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                const int srcLane = k * BS + cc;                     // 同列第 k 行的 lane
                const float rhs_kc = __shfl_sync(act, rhs_rc, srcLane);
                y += inv[rr * BS + k] * rhs_kc;
            }
            // 写回
            Mval[(start + idx) * BS2 + lane] = y;
        }
        __syncwarp(act);
    }
}

// 3x3 打包：一次 WMMA 累出 9 对 A*Yp 的和（取 C 的三块 5x5 对角相加）

// npairs > 4 用 3×3 组块一次 WMMA；否则标量 5×5
/*__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_tc_thresh4(
    int nb,
    const int*  __restrict__ Abrow,   // 保留接口（未用）
    const int*  __restrict__ Abcol,   // 保留接口（未用）
    const float*__restrict__ Abval,   // A 的块值 (float), 每块 25
    const int*  __restrict__ Mbrow,   // per-column rowptr
    const int*  __restrict__ Mcol,    // S_j
    const int*  __restrict__ PairPtr, // 对列表 rowptr
    const int*  __restrict__ PairPLocal,// p 的列内局部号
    const int*  __restrict__ PairSrc, // A 中 (t,p) 的 src_pos（Abval 的块号）
    float*      __restrict__ Mval,    // 输出（BCSC 顺序）
    const float*__restrict__ DinvL    // 预逆 (nb*25)
){
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    const int j = blockIdx.x * warpsPerBlock + warp; // warp -> column
    if (j >= nb) return;

    const int start = Mbrow[j];
    const int end   = Mbrow[j+1];
    const int N     = end - start;
    if (N <= 0) return;

    // 每 warp: 3*16*16 的共享缓冲 (A16/B16/C16)
    extern __shared__ float sh[];
    float* sA16 = sh + warp * (3 * 16 * 16) + 0 * 16 * 16;
    float* sB16 = sh + warp * (3 * 16 * 16) + 1 * 16 * 16;
    float* sC16 = sh + warp * (3 * 16 * 16) + 2 * 16 * 16;

    const int rr = lane / BS;   // 0..4
    const int cc = lane % BS;   // 0..4

    const unsigned FULL = 0xFFFFFFFFu;
    const unsigned act  = __ballot_sync(FULL, lane < BS2);

    for (int idx = 0; idx < N; ++idx) {
        const int t = Mcol[start + idx];

        // rhs 寄存器化
        float rhs_rc = (lane < BS2 && t == j && rr == cc) ? 1.f : 0.f;

        const int g0     = start + idx;
        const int qbeg   = PairPtr[g0];
        const int qend   = PairPtr[g0 + 1];
        const int npairs = qend - qbeg;

        float acc_rc = 0.f;  // 累计 sum_q A*Yp 的 (rr,cc)

        if (npairs > 4) {
            // ---------------- 3×3 打包（一次 WMMA 累出最多 9 对）----------------
            for (int qb = qbeg; qb < qend; qb += 9) {
                const int k = min(9, qend - qb);

                // （1）清零 16×16 A/B（本批只清一次）
                for (int t2 = lane; t2 < 16*16; t2 += 32) {
                    sA16[t2] = 0.f;
                    sB16[t2] = 0.f;
                }
                __syncwarp(act);

                // （2）把本批 k 个 pair 写入 3×3 网格：A 放 (i,j)，B 放 (j,i)
                #pragma unroll
                for (int m = 0; m < 9; ++m) {
                    if (m >= k) break;
                    const int q     = qb + m;
                    const int p_loc = PairPLocal[q];
                    const int srcA  = PairSrc   [q];

                    const int bi = m / 3;   // 0..2
                    const int bj = m % 3;   // 0..2

                    const int aro = bi * BS, aco = bj * BS;  // A 的 (i,j) 偏移
                    const int bro = bj * BS, bco = bi * BS;  // B 的 (j,i) 偏移

                    for (int e = lane; e < BS2; e += 32) {
                        const int r = e / BS, c = e % BS;
                        sA16[(aro + r) * 16 + (aco + c)] = Abval[srcA * BS2 + e];
                        sB16[(bro + r) * 16 + (bco + c)] = Mval[(start + p_loc) * BS2 + e];
                    }
                    __syncwarp(act);
                }

                // （3）K=16 拆成两段 m16n16k8（TF32）
                wmma::fragment<wmma::accumulator,       16,16,8, float> Cf;
                wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::row_major> Af;
                wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::row_major> Bf;

                wmma::fill_fragment(Cf, 0.0f);
                // K=0..7
                wmma::load_matrix_sync(Af, sA16 + 0,    16);
                wmma::load_matrix_sync(Bf, sB16 + 0*16, 16);
                wmma::mma_sync(Cf, Af, Bf, Cf);
                // K=8..15
                wmma::load_matrix_sync(Af, sA16 + 8,    16);
                wmma::load_matrix_sync(Bf, sB16 + 8*16, 16);
                wmma::mma_sync(Cf, Af, Bf, Cf);

                wmma::store_matrix_sync(sC16, Cf, 16, wmma::mem_row_major);
                __syncwarp(act);

                // （4）取 C 的三块 5×5 对角（0,0),(5,5),(10,10) 相加
                if (lane < BS2) {
                    acc_rc += sC16[(0  + rr) * 16 + (0  + cc)];
                    acc_rc += sC16[(5  + rr) * 16 + (5  + cc)];
                    acc_rc += sC16[(10 + rr) * 16 + (10 + cc)];
                }
                __syncwarp(act);
            }
        } else {
            // ---------------- 标量 5×5 路径（npairs ≤ 4）----------------
            for (int q = qbeg; q < qend; ++q) {
                const int p_loc = PairPLocal[q];
                const int srcA  = PairSrc   [q];
                if (lane < BS2) {
                    // sum_k A(rr,k) * Yp(k,cc)
                    float sum = 0.f;
                    #pragma unroll
                    for (int k2 = 0; k2 < BS; ++k2) {
                        const float a = Abval[srcA * BS2 + rr * BS + k2];
                        const float y = Mval[(start + p_loc) * BS2 + k2 * BS + cc];
                        sum += a * y;
                    }
                    acc_rc += sum;
                }
                __syncwarp(act);
            }
        }

        // 一次性 rhs -= 累计和
        if (lane < BS2) rhs_rc -= acc_rc;
        __syncwarp(act);

        // ---- 标量 5×5：Y = DinvL[t] * rhs（列内广播）----
        if (lane < BS2) {
            const float* inv = DinvL + t * BS2;
            float y = 0.f;
            #pragma unroll
            for (int k2 = 0; k2 < BS; ++k2) {
                const int srcLane = k2 * BS + cc;         // 同列第 k2 行
                const float rhs_kc = __shfl_sync(act, rhs_rc, srcLane);
                y += inv[rr * BS + k2] * rhs_kc;
            }
            Mval[(start + idx) * BS2 + lane] = y;
        }
        __syncwarp(act);
    }
}
*/

// Tensor Core：仅用于 A*Yp；sRhs 寄存器化；DinvL*rhs 标量 5×5；两对一批（TL/BR）

// 阈值：npairs>4 才用 FP16 (m16n16k16) 的 WMMA，两对一批（TL/BR）；否则标量 5x5
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_tc_fp16_thresh4(
    int nb,
    const int*  __restrict__ Abrow,   // 保留接口（未用）
    const int*  __restrict__ Abcol,   // 保留接口（未用）
    const float*__restrict__ Abval,   // A 的块值 (float), 每块 25
    const int*  __restrict__ Mbrow,   // per-column rowptr
    const int*  __restrict__ Mcol,    // S_j
    const int*  __restrict__ PairPtr, // 对列表 rowptr
    const int*  __restrict__ PairPLocal,// p 的列内局部号
    const int*  __restrict__ PairSrc, // A 中 (t,p) 的 src_pos（Abval 的块号）
    float*      __restrict__ Mval,    // 输出（BCSC 顺序，float）
    const float*__restrict__ DinvL    // 预逆 (nb*25，float)
){
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    const int j = blockIdx.x * warpsPerBlock + warp; // warp -> column
    if (j >= nb) return;

    const int start = Mbrow[j];
    const int end   = Mbrow[j+1];
    const int N     = end - start;
    if (N <= 0) return;

    // --- 每 warp 的共享内存：A16h/B16h（half）、C16f（float） ---
    extern __shared__ unsigned char raw[];
    // 128B 对齐
    auto align128 = [](size_t x){ return (x + 127u) & ~127u; };
    const size_t offA = 0;
    const size_t offB = align128(offA + 16*16*sizeof(__half));
    const size_t offC = align128(offB + 16*16*sizeof(__half));
    __half* sA16h = reinterpret_cast<__half*>(raw + warp * (offC + 16*16*sizeof(float)) + offA);
    __half* sB16h = reinterpret_cast<__half*>(raw + warp * (offC + 16*16*sizeof(float)) + offB);
    float*  sC16f = reinterpret_cast<float* >(raw + warp * (offC + 16*16*sizeof(float)) + offC);

    const int rr = lane / BS;   // 0..4
    const int cc = lane % BS;   // 0..4

    const unsigned FULL = 0xFFFFFFFFu;
    const unsigned act  = __ballot_sync(FULL, lane < BS2);

    for (int idx = 0; idx < N; ++idx) {
        const int t = Mcol[start + idx];

        // rhs 寄存器（单位列）
        float rhs_rc = (lane < BS2 && t == j && rr == cc) ? 1.f : 0.f;

        const int g0     = start + idx;
        const int qbeg   = PairPtr[g0];
        const int qend   = PairPtr[g0 + 1];
        const int npairs = qend - qbeg;

        float acc_rc = 0.f;  // 累计 sum_q A*Yp 的 (rr,cc)

        if (npairs > 2) {
            // ---- 两对一批：TL/BR，一次 m16n16k16 ----
            int qb = qbeg;
            for (; qb + 1 < qend; qb += 2) {
                // 清零 tile（half）
                for (int t2 = lane; t2 < 16*16; t2 += 32) {
                    sA16h[t2] = __float2half(0.f);
                    sB16h[t2] = __float2half(0.f);
                }
                __syncwarp(act);

                // 对0 -> TL(0,0)
                {
                    const int p0 = PairPLocal[qb + 0];
                    const int s0 = PairSrc   [qb + 0];
                    for (int e = lane; e < BS2; e += 32) {
                        const int r = e / BS, c = e % BS;
                        const float a = Abval[s0 * BS2 + e];
                        const float y = Mval [(start + p0) * BS2 + e];
                        sA16h[(0 + r)*16 + (0 + c)] = __float2half_rn(a);
                        sB16h[(0 + r)*16 + (0 + c)] = __float2half_rn(y);
                    }
                    __syncwarp(act);
                }
                // 对1 -> BR(8,8)
                {
                    const int p1 = PairPLocal[qb + 1];
                    const int s1 = PairSrc   [qb + 1];
                    for (int e = lane; e < BS2; e += 32) {
                        const int r = e / BS, c = e % BS;
                        const float a = Abval[s1 * BS2 + e];
                        const float y = Mval [(start + p1) * BS2 + e];
                        sA16h[(8 + r)*16 + (8 + c)] = __float2half_rn(a);
                        sB16h[(8 + r)*16 + (8 + c)] = __float2half_rn(y);
                    }
                    __syncwarp(act);
                }

                // 1 次 WMMA：m16n16k16（half → float 累加）
                wmma::fragment<wmma::accumulator, 16,16,16, float> Cf;
                wmma::fragment<wmma::matrix_a,    16,16,16, half,  wmma::row_major> Af;
                wmma::fragment<wmma::matrix_b,    16,16,16, half,  wmma::row_major> Bf;

                wmma::fill_fragment(Cf, 0.0f);
                wmma::load_matrix_sync(Af, sA16h, 16);
                wmma::load_matrix_sync(Bf, sB16h, 16);
                wmma::mma_sync(Cf, Af, Bf, Cf);
                wmma::store_matrix_sync(sC16f, Cf, 16, wmma::mem_row_major);
                __syncwarp(act);

                // TL + BR 的 5x5 累加
                if (lane < BS2) {
                    acc_rc += sC16f[rr*16 + cc];
                    acc_rc += sC16f[(8 + rr)*16 + (8 + cc)];
                }
                __syncwarp(act);
            }

            // 零头（奇数 1 对）→ 标量 5×5
            if (qb < qend) {
                const int p = PairPLocal[qb];
                const int s = PairSrc   [qb];
                if (lane < BS2) {
                    float sum = 0.f;
                    #pragma unroll
                    for (int k2 = 0; k2 < BS; ++k2) {
                        const float a = Abval[s * BS2 + rr * BS + k2];
                        const float y = Mval [(start + p) * BS2 + k2 * BS + cc];
                        sum += a * y;
                    }
                    acc_rc += sum;
                }
                __syncwarp(act);
            }
        } else {
            // ---- 标量 5×5：npairs ≤ 4 ----
            for (int q = qbeg; q < qend; ++q) {
                const int p = PairPLocal[q];
                const int s = PairSrc   [q];
                if (lane < BS2) {
                    float sum = 0.f;
                    #pragma unroll
                    for (int k2 = 0; k2 < BS; ++k2) {
                        const float a = Abval[s * BS2 + rr * BS + k2];
                        const float y = Mval [(start + p) * BS2 + k2 * BS + cc];
                        sum += a * y;
                    }
                    acc_rc += sum;
                }
                __syncwarp(act);
            }
        }

        // rhs -= 累计和
        if (lane < BS2) rhs_rc -= acc_rc;
        __syncwarp(act);

        // ---- 标量 5×5：Y = DinvL[t]*rhs（FP32 累加，warp 内列广播）----
        if (lane < BS2) {
            const float* inv = DinvL + t * BS2;
            float y = 0.f;
            #pragma unroll
            for (int k2 = 0; k2 < BS; ++k2) {
                const int srcLane = k2 * BS + cc;
                const float rhs_kc = __shfl_sync(act, rhs_rc, srcLane);
                y += inv[rr*BS + k2] * rhs_kc;
            }
            Mval[(start + idx) * BS2 + lane] = y;
        }
        __syncwarp(act);
    }
}



// 3×3 打包（15×15→16×16）：一次 m16n16k16 (half->float)；取 C 的三块 5×5 对角相加

// npairs>4: 3×3 打包（15×15→16×16），一次 FP16 m16n16k16；否则：标量 5×

// npairs>4: 3×3 组块（15×15→16×16），一次 FP16 m16n16k16；
// npairs<=4: 对角“两对一批”Tensor（TL/BR），剩余1对用单TL。

// npairs>4: 3×3 组块（15×15→16×16），一次 FP16 m16n16k16；
// npairs<=4: 对角两对一批 Tensor（TL/BR），若剩余1对 -> 标量 5×5。
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_tc_hybrid_scalar_tail(
    int nb,
    const int*  __restrict__ Abrow,     // 保留接口（未用）
    const int*  __restrict__ Abcol,     // 保留接口（未用）
    const float*__restrict__ Abval,     // A 的块值 (float), 每块25
    const int*  __restrict__ Mbrow,     // per-column rowptr
    const int*  __restrict__ Mcol,      // S_j
    const int*  __restrict__ PairPtr,   // 对列表 rowptr（与 Mbrow/Mcol 对齐）
    const int*  __restrict__ PairPLocal,// p 的列内局部号
    const int*  __restrict__ PairSrc,   // A 中 (t,p) 的 src_pos（Abval 的块号）
    float*      __restrict__ Mval,      // 输出（BCSC 顺序）
    const float*__restrict__ DinvL      // 预逆 (nb*25)
){
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    const int j = blockIdx.x * warpsPerBlock + warp; // warp -> column
    if (j >= nb) return;

    const int start = Mbrow[j];
    const int end   = Mbrow[j+1];
    const int N     = end - start;
    if (N <= 0) return;

    // 每 warp 共享：A16h / B16h / C16f（用 float 空间承载，简单稳定）
    extern __shared__ float shmem[];
    float* warp_base = shmem + warp * (3 * 16 * 16);
    half*  sA16h = reinterpret_cast<half*>( warp_base + 0 * 16 * 16 );
    half*  sB16h = reinterpret_cast<half*>( warp_base + 1 * 16 * 16 );
    float* sC16f =                ( warp_base + 2 * 16 * 16 );

    const int rr = lane / BS;   // 0..4
    const int cc = lane % BS;   // 0..4

    const unsigned FULL = 0xFFFFFFFFu;
    const unsigned act  = __ballot_sync(FULL, lane < BS2);

    for (int idx = 0; idx < N; ++idx) {
        const int t = Mcol[start + idx];

        // rhs 寄存器化（单位列）
        float rhs_rc = (lane < BS2 && t == j && rr == cc) ? 1.f : 0.f;

        const int g0     = start + idx;
        const int qbeg   = PairPtr[g0];
        const int qend   = PairPtr[g0 + 1];
        const int npairs = qend - qbeg;

        float acc_rc = 0.f;  // 累计 sum_q A*Yp 的 (rr,cc)

        if (npairs > 4) {
            // ---------------- 3×3 组块：一次 WMMA 最多吃 9 对 ----------------
            for (int qb = qbeg; qb < qend; qb += 9) {
                const int k = min(9, qend - qb);

                // (1) 清零 16×16（A/B）
                for (int t2 = lane; t2 < 16*16; t2 += 32) {
                    sA16h[t2] = __float2half_rn(0.f);
                    sB16h[t2] = __float2half_rn(0.f);
                }
                __syncwarp(act);

                // (2) 把 k 个 pair 以 3×3 放入：A 放 (i,j)，B 放 (j,i)
                #pragma unroll
                for (int m = 0; m < 9; ++m) {
                    if (m >= k) break;
                    const int q     = qb + m;
                    const int p_loc = PairPLocal[q];
                    const int srcA  = PairSrc   [q];

                    const int bi = m / 3;  // 0..2
                    const int bj = m % 3;  // 0..2

                    const int aro = bi * BS; // A 行偏移
                    const int aco = bj * BS; // A 列偏移
                    const int bro = bj * BS; // B 行偏移（转置）
                    const int bco = bi * BS; // B 列偏移（转置）

                    for (int e = lane; e < BS2; e += 32) {
                        const int r = e / BS, c = e % BS;
                        const float a = Abval[srcA * BS2 + e];
                        const float y = Mval [(start + p_loc) * BS2 + e];
                        sA16h[(aro + r) * 16 + (aco + c)] = __float2half_rn(a);
                        sB16h[(bro + r) * 16 + (bco + c)] = __float2half_rn(y);
                    }
                    __syncwarp(act);
                }

                // (3) 1 次 WMMA：m16n16k16（half 输入，float 累加）
                wmma::fragment<wmma::accumulator, 16,16,16, float> Cf;
                wmma::fragment<wmma::matrix_a,    16,16,16, half,  wmma::row_major> Af;
                wmma::fragment<wmma::matrix_b,    16,16,16, half,  wmma::row_major> Bf;

                wmma::fill_fragment(Cf, 0.0f);
                wmma::load_matrix_sync(Af, sA16h, 16);
                wmma::load_matrix_sync(Bf, sB16h, 16);
                wmma::mma_sync(Cf, Af, Bf, Cf);
                wmma::store_matrix_sync(sC16f, Cf, 16, wmma::mem_row_major);
                __syncwarp(act);

                // (4) 取 C 的三块 5×5 对角（(0,0),(5,5),(10,10)）相加
                if (lane < BS2) {
                    acc_rc += sC16f[(0  + rr) * 16 + (0  + cc)];
                    acc_rc += sC16f[(5  + rr) * 16 + (5  + cc)];
                    acc_rc += sC16f[(10 + rr) * 16 + (10 + cc)];
                }
                __syncwarp(act);
            }
        } else {
            // --------------- 对角两对一批 Tensor（TL/BR） ---------------
            int qb = qbeg;
            for (; qb + 1 < qend; qb += 2) {
                // 清零 A/B tile
                for (int t2 = lane; t2 < 16*16; t2 += 32) {
                    sA16h[t2] = __float2half_rn(0.f);
                    sB16h[t2] = __float2half_rn(0.f);
                }
                __syncwarp(act);

                // 对0 -> TL(0,0)
                {
                    const int p0 = PairPLocal[qb + 0];
                    const int s0 = PairSrc   [qb + 0];
                    for (int e = lane; e < BS2; e += 32) {
                        const int r = e / BS, c = e % BS;
                        const float a = Abval[s0 * BS2 + e];
                        const float y = Mval [(start + p0) * BS2 + e];
                        sA16h[(0 + r)*16 + (0 + c)] = __float2half_rn(a);
                        sB16h[(0 + r)*16 + (0 + c)] = __float2half_rn(y);
                    }
                    __syncwarp(act);
                }
                // 对1 -> BR(8,8)
                {
                    const int p1 = PairPLocal[qb + 1];
                    const int s1 = PairSrc   [qb + 1];
                    for (int e = lane; e < BS2; e += 32) {
                        const int r = e / BS, c = e % BS;
                        const float a = Abval[s1 * BS2 + e];
                        const float y = Mval [(start + p1) * BS2 + e];
                        sA16h[(8 + r)*16 + (8 + c)] = __float2half_rn(a);
                        sB16h[(8 + r)*16 + (8 + c)] = __float2half_rn(y);
                    }
                    __syncwarp(act);
                }

                // 1 次 WMMA：m16n16k16（half→float）
                wmma::fragment<wmma::accumulator, 16,16,16, float> Cf2;
                wmma::fragment<wmma::matrix_a,    16,16,16, half,  wmma::row_major> Af2;
                wmma::fragment<wmma::matrix_b,    16,16,16, half,  wmma::row_major> Bf2;

                wmma::fill_fragment(Cf2, 0.0f);
                wmma::load_matrix_sync(Af2, sA16h, 16);
                wmma::load_matrix_sync(Bf2, sB16h, 16);
                wmma::mma_sync(Cf2, Af2, Bf2, Cf2);
                wmma::store_matrix_sync(sC16f, Cf2, 16, wmma::mem_row_major);
                __syncwarp(act);

                // TL + BR 的 5×5 累加
                if (lane < BS2) {
                    acc_rc += sC16f[rr*16 + cc];                    // TL
                    acc_rc += sC16f[(8 + rr)*16 + (8 + cc)];        // BR
                }
                __syncwarp(act);
            }

            // --------------- 零头（1 对）→ 标量 5×5 ---------------
            if (qb < qend) {
                const int p = PairPLocal[qb];
                const int s = PairSrc   [qb];
                if (lane < BS2) {
                    float sum = 0.f;
                    #pragma unroll
                    for (int k2 = 0; k2 < BS; ++k2) {
                        const float a = Abval[s * BS2 + rr * BS + k2];
                        const float y = Mval [(start + p) * BS2 + k2 * BS + cc];
                        sum += a * y;
                    }
                    acc_rc += sum;
                }
                __syncwarp(act);
            }
        }

        // 累减 rhs
        if (lane < BS2) rhs_rc -= acc_rc;
        __syncwarp(act);

        // Y = DinvL[t] * rhs（标量 5×5，warp 内列广播）
        if (lane < BS2) {
            const float* inv = DinvL + t * BS2;
            float y = 0.f;
            #pragma unroll
            for (int k2 = 0; k2 < BS; ++k2) {
                const int srcLane = k2 * BS + cc;
                const float rhs_kc = __shfl_sync(act, rhs_rc, srcLane);
                y += inv[rr * BS + k2] * rhs_kc;
            }
            Mval[(start + idx) * BS2 + lane] = y;
        }
        __syncwarp(act);
    }
}


__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_tc_two_per_batch(
    int nb,
    const int*  __restrict__ Abrow,     // 保留接口（未用）
    const int*  __restrict__ Abcol,     // 保留接口（未用）
    const float*__restrict__ Abval,     // A 的块值 (float), 每块25
    const int*  __restrict__ Mbrow,     // per-column rowptr
    const int*  __restrict__ Mcol,      // S_j
    const int*  __restrict__ PairPtr,   // 对列表 rowptr（与 Mbrow/Mcol 对齐）
    const int*  __restrict__ PairPLocal,// p 的列内局部号
    const int*  __restrict__ PairSrc,   // A 中 (t,p) 的 src_pos（Abval 的块号）
    float*      __restrict__ Mval,      // 输出（BCSC 顺序）
    const float*__restrict__ DinvL      // 预逆 (nb*25)
){
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    const int j = blockIdx.x * warpsPerBlock + warp; // warp -> column
    if (j >= nb) return;

    const int start = Mbrow[j];
    const int end   = Mbrow[j+1];
    const int N     = end - start;
    if (N <= 0) return;

    // 每 warp 共享：A16h / B16h / C16f（用 float 空间承载，简单稳定）
    extern __shared__ float shmem[];
    float* warp_base = shmem + warp * (3 * 16 * 16);
    half*  sA16h = reinterpret_cast<half*>( warp_base + 0 * 16 * 16 );
    half*  sB16h = reinterpret_cast<half*>( warp_base + 1 * 16 * 16 );
    float* sC16f =                ( warp_base + 2 * 16 * 16 );

    const int rr = lane / BS;   // 0..4（行）
    const int cc = lane % BS;   // 0..4（列）

    const unsigned FULL = 0xFFFFFFFFu;
    const unsigned act  = __ballot_sync(FULL, lane < BS2);

    for (int idx = 0; idx < N; ++idx) {
        const int t = Mcol[start + idx];

        // rhs 寄存器化（单位列）
        float rhs_rc = (lane < BS2 && t == j && rr == cc) ? 1.f : 0.f;

        const int g0     = start + idx;
        const int qbeg   = PairPtr[g0];
        const int qend   = PairPtr[g0 + 1];

        float acc_rc = 0.f;  // 累计 sum_q A*Yp 的 (rr,cc)

        // -------- 两对一批：pair0->TL, pair1->BR --------
        int qb = qbeg;
        for (; qb + 1 < qend; qb += 2) {
            // 清零 A/B tile（half）
            for (int t2 = lane; t2 < 16*16; t2 += 32) {
                sA16h[t2] = __float2half_rn(0.f);
                sB16h[t2] = __float2half_rn(0.f);
            }
            __syncwarp(act);

            // 对0 -> TL(0,0)
            {
                const int p0 = PairPLocal[qb + 0];
                const int s0 = PairSrc   [qb + 0];
                for (int e = lane; e < BS2; e += 32) {
                    const int r = e / BS, c = e % BS;
                    const float a = Abval[s0 * BS2 + e];
                    const float y = Mval [(start + p0) * BS2 + e];
                    sA16h[(0 + r)*16 + (0 + c)] = __float2half_rn(a);
                    sB16h[(0 + r)*16 + (0 + c)] = __float2half_rn(y);
                }
                __syncwarp(act);
            }
            // 对1 -> BR(8,8)
            {
                const int p1 = PairPLocal[qb + 1];
                const int s1 = PairSrc   [qb + 1];
                for (int e = lane; e < BS2; e += 32) {
                    const int r = e / BS, c = e % BS;
                    const float a = Abval[s1 * BS2 + e];
                    const float y = Mval [(start + p1) * BS2 + e];
                    sA16h[(8 + r)*16 + (8 + c)] = __float2half_rn(a);
                    sB16h[(8 + r)*16 + (8 + c)] = __float2half_rn(y);
                }
                __syncwarp(act);
            }

            // 一次 WMMA：m16n16k16（half→float）
            wmma::fragment<wmma::accumulator, 16,16,16, float> Cf;
            wmma::fragment<wmma::matrix_a,    16,16,16, half,  wmma::row_major> Af;
            wmma::fragment<wmma::matrix_b,    16,16,16, half,  wmma::row_major> Bf;

            wmma::fill_fragment(Cf, 0.0f);
            wmma::load_matrix_sync(Af, sA16h, 16);
            wmma::load_matrix_sync(Bf, sB16h, 16);
            wmma::mma_sync(Cf, Af, Bf, Cf);
            wmma::store_matrix_sync(sC16f, Cf, 16, wmma::mem_row_major);
            __syncwarp(act);

            // TL + BR 的 5×5 累加
            if (lane < BS2) {
                acc_rc += sC16f[rr*16 + cc];                    // TL
                acc_rc += sC16f[(8 + rr)*16 + (8 + cc)];        // BR
            }
            __syncwarp(act);
        }

        // -------- 零头（1 对）：只填 TL，再做一次 WMMA --------
        if (qb < qend) {
            // 清零
            for (int t2 = lane; t2 < 16*16; t2 += 32) {
                sA16h[t2] = __float2half_rn(0.f);
                sB16h[t2] = __float2half_rn(0.f);
            }
            __syncwarp(act);

            const int p = PairPLocal[qb];
            const int s = PairSrc   [qb];
            for (int e = lane; e < BS2; e += 32) {
                const int r = e / BS, c = e % BS;
                const float a = Abval[s * BS2 + e];
                const float y = Mval [(start + p) * BS2 + e];
                sA16h[(0 + r)*16 + (0 + c)] = __float2half_rn(a);
                sB16h[(0 + r)*16 + (0 + c)] = __float2half_rn(y);
            }
            __syncwarp(act);

            wmma::fragment<wmma::accumulator, 16,16,16, float> Cf2;
            wmma::fragment<wmma::matrix_a,    16,16,16, half,  wmma::row_major> Af2;
            wmma::fragment<wmma::matrix_b,    16,16,16, half,  wmma::row_major> Bf2;

            wmma::fill_fragment(Cf2, 0.0f);
            wmma::load_matrix_sync(Af2, sA16h, 16);
            wmma::load_matrix_sync(Bf2, sB16h, 16);
            wmma::mma_sync(Cf2, Af2, Bf2, Cf2);
            wmma::store_matrix_sync(sC16f, Cf2, 16, wmma::mem_row_major);
            __syncwarp(act);

            if (lane < BS2) {
                acc_rc += sC16f[rr*16 + cc]; // TL
            }
            __syncwarp(act);
        }

        // rhs -= 累计和
        if (lane < BS2) rhs_rc -= acc_rc;
        __syncwarp(act);

        // Y = DinvL[t] * rhs（标量 5×5，warp 内列广播）
        if (lane < BS2) {
            const float* inv = DinvL + t * BS2;
            float y = 0.f;
            #pragma unroll
            for (int k2 = 0; k2 < BS; ++k2) {
                const int srcLane = k2 * BS + cc;   // 同列第 k2 行
                const float rhs_kc = __shfl_sync(act, rhs_rc, srcLane);
                y += inv[rr * BS + k2] * rhs_kc;
            }
            Mval[(start + idx) * BS2 + lane] = y;
        }
        __syncwarp(act);
    }
}


// 两对一批 Tensor Core（FP16 m16n16k16），带：
// - Cf 复用多批累加（只在末尾 store 一次）
// - 只在本 idx 开头清整 tile，其余批次只覆盖 TL/BR 的 5x5
// - float4 装填（16B 对齐才用，否则回退标量）
// - 列内 Y 缓存（shared），超出 Nmax_perwarp 时自动回退为 global 读
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_tc_two_per_batch_optim(
    int nb,
    const int*  __restrict__ Abrow,     // 保留接口（未用）
    const int*  __restrict__ Abcol,     // 保留接口（未用）
    const float*__restrict__ Abval,     // A 的块值 (float), 每块25，行主序
    const int*  __restrict__ Mbrow,     // per-column rowptr
    const int*  __restrict__ Mcol,      // S_j
    const int*  __restrict__ PairPtr,   // 对列表 rowptr（与 Mbrow/Mcol 对齐）
    const int*  __restrict__ PairPLocal,// p 的列内局部号（< idx）
    const int*  __restrict__ PairSrc,   // A 中 (t,p) 的 src_pos（Abval 的块号）
    float*      __restrict__ Mval,      // 输出（BCSC 顺序），每块25
    const float*__restrict__ DinvL,     // 预逆 (nb*25，float)
    int Nmax_perwarp                     // 列内缓存的最大块数（例如 128）
){
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    const int j = blockIdx.x * warpsPerBlock + warp; // warp -> column
    if (j >= nb) return;

    const int start = Mbrow[j];
    const int end   = Mbrow[j+1];
    const int N     = end - start;
    if (N <= 0) return;
    // ---------------- 共享内存布局（按 warp 切片 + 128B 对齐） ----------------
    extern __shared__ __align__(16) unsigned char sbytes[];
    float* smem = reinterpret_cast<float*>(sbytes);
             // 用 float 计量，便于对齐
    const int ALIGN_WORDS = 128 / sizeof(float);

    const int words_sY   = min(N, Nmax_perwarp) * BS2;      // 列内 Y 缓存
    int words_base       = words_sY;
    int pad0             = (ALIGN_WORDS - (words_base % ALIGN_WORDS)) & (ALIGN_WORDS - 1);
    const int words_tiles= 3 * 16 * 16;                     // A16h(当作float区) + B16h + C16f
    const int per_warp_words = words_base + pad0 + words_tiles;

    float* warp_base = smem + warp * per_warp_words;

    // 列内缓存
    float* sYall = warp_base;                                // [min(N,Nmax)*25]
    // 16x16 瓦片缓冲（用 float 空间承载，前两块 reinterpret_cast 为 half*）
    float* tiles = sYall + words_sY + pad0;
    half*  sA16h = reinterpret_cast<half*>( tiles + 0 * 16 * 16 ); // [256 half]
    half*  sB16h = reinterpret_cast<half*>( tiles + 1 * 16 * 16 ); // [256 half]
    float* sC16f =                ( tiles + 2 * 16 * 16 );         // [256 float]

    const bool use_cache = (N <= Nmax_perwarp);

    const int rr = lane / BS;   // 0..4
    const int cc = lane % BS;   // 0..4

    const unsigned FULL = 0xFFFFFFFFu;
    const unsigned act  = __ballot_sync(FULL, lane < BS2);

    // ---------------- 小工具：把 5x5 从 src(float) 写到 tile(half) 的 (r0,c0) ----------------
    auto write5x5_half = [&](const float* __restrict__ src, half* __restrict__ tile, int r0, int c0){
        // 用 5 个 lane（0..4）各写一整行；对齐则用 float4，否则标量
        if (lane < BS) {
            int r = lane;                  // 行
            const float* row = src + r * BS;
            // 取前4个元素
            float a0, a1, a2, a3, a4;
            uintptr_t addr = reinterpret_cast<uintptr_t>(row);
            if ((addr & 0xF) == 0) {
                // 16B 对齐：float4 装载
                float4 v4 = *reinterpret_cast<const float4*>(row);
                a0 = v4.x; a1 = v4.y; a2 = v4.z; a3 = v4.w;
                a4 = row[4];
            } else {
                // 回退标量
                a0 = row[0]; a1 = row[1]; a2 = row[2]; a3 = row[3]; a4 = row[4];
            }
            int base = (r0 + r) * 16 + c0;
            tile[base + 0] = __float2half_rn(a0);
            tile[base + 1] = __float2half_rn(a1);
            tile[base + 2] = __float2half_rn(a2);
            tile[base + 3] = __float2half_rn(a3);
            tile[base + 4] = __float2half_rn(a4);
        }
        __syncwarp(act);
    };

    // 清 5x5（半精）
    auto zero5x5_half = [&](half* __restrict__ tile, int r0, int c0){
        if (lane < BS) {
            int r = lane;
            int base = (r0 + r) * 16 + c0;
            tile[base + 0] = __float2half_rn(0.f);
            tile[base + 1] = __float2half_rn(0.f);
            tile[base + 2] = __float2half_rn(0.f);
            tile[base + 3] = __float2half_rn(0.f);
            tile[base + 4] = __float2half_rn(0.f);
        }
        __syncwarp(act);
    };

    for (int idx = 0; idx < N; ++idx) {
        const int t = Mcol[start + idx];

        // rhs（单位列）寄存器化
        float rhs_rc = (lane < BS2 && t == j && rr == cc) ? 1.f : 0.f;

        const int g0   = start + idx;
        const int qbeg = PairPtr[g0];
        const int qend = PairPtr[g0 + 1];

        // -------------- 初始化：整 tile 清 0（仅一次/idx），Cf 清 0 --------------
        for (int off = lane; off < 16*16; off += 32) {
            sA16h[off] = __float2half_rn(0.f);
            sB16h[off] = __float2half_rn(0.f);
        }
        __syncwarp(act);

        wmma::fragment<wmma::accumulator, 16,16,16, float> Cf;
        wmma::fragment<wmma::matrix_a,    16,16,16, half,  wmma::row_major> Af;
        wmma::fragment<wmma::matrix_b,    16,16,16, half,  wmma::row_major> Bf;
        wmma::fill_fragment(Cf, 0.0f);

        // -------------- 遍历所有 pair：两对一批，Cf 累加，不 store --------------
        int qb = qbeg;
        for (; qb + 1 < qend; qb += 2) {
            // pair0 -> TL(0,0)
            {
                const int p0 = PairPLocal[qb + 0];
                const int s0 = PairSrc   [qb + 0];
                const float* A0 = Abval + s0 * BS2;
                const float* Y0 = (use_cache ? (sYall + p0 * BS2) : (Mval + (start + p0) * BS2));
                write5x5_half(A0, sA16h, 0, 0);
                write5x5_half(Y0, sB16h, 0, 0);
            }
            // pair1 -> BR(8,8)
            {
                const int p1 = PairPLocal[qb + 1];
                const int s1 = PairSrc   [qb + 1];
                const float* A1 = Abval + s1 * BS2;
                const float* Y1 = (use_cache ? (sYall + p1 * BS2) : (Mval + (start + p1) * BS2));
                write5x5_half(A1, sA16h, 8, 8);
                write5x5_half(Y1, sB16h, 8, 8);
            }

            // 累加到 Cf（不 store）
            wmma::load_matrix_sync(Af, sA16h, 16);
            wmma::load_matrix_sync(Bf, sB16h, 16);
            wmma::mma_sync(Cf, Af, Bf, Cf);
            __syncwarp(act);
        }

        // 零头（1 对）：TL 写入，BR 显式清 5x5，再累加一次
        if (qb < qend) {
            const int p = PairPLocal[qb];
            const int s = PairSrc   [qb];
            const float* A = Abval + s * BS2;
            const float* Y = (use_cache ? (sYall + p * BS2) : (Mval + (start + p) * BS2));
            write5x5_half(A, sA16h, 0, 0);
            write5x5_half(Y, sB16h, 0, 0);
            zero5x5_half(sA16h, 8, 8);
            zero5x5_half(sB16h, 8, 8);

            wmma::load_matrix_sync(Af, sA16h, 16);
            wmma::load_matrix_sync(Bf, sB16h, 16);
            wmma::mma_sync(Cf, Af, Bf, Cf);
            __syncwarp(act);
        }

        // -------------- 只在末尾 store 一次，然后读 TL/BR 5x5 累到 acc_rc --------------
        wmma::store_matrix_sync(sC16f, Cf, 16, wmma::mem_row_major);
        __syncwarp(act);

        float acc_rc = 0.f;
        if (lane < BS2) {
            acc_rc += sC16f[rr*16 + cc];                    // TL
            acc_rc += sC16f[(8 + rr)*16 + (8 + cc)];        // BR
        }
        __syncwarp(act);

        // rhs -= 累计和
        if (lane < BS2) rhs_rc -= acc_rc;
        __syncwarp(act);

        // -------------- 标量 5x5：Y = DinvL[t] * rhs（列内广播） --------------
        if (lane < BS2) {
            const float* inv = DinvL + t * BS2;
            float y = 0.f;
            #pragma unroll
            for (int k2 = 0; k2 < BS; ++k2) {
                const int srcLane = k2 * BS + cc;   // 同列第 k2 行
                const float rhs_kc = __shfl_sync(act, rhs_rc, srcLane);
                y += inv[rr * BS + k2] * rhs_kc;
            }
            // 写回全局 + 写入共享缓存
            Mval[(start + idx) * BS2 + lane] = y;
            if (use_cache) sYall[idx * BS2 + lane] = y;
        }
        __syncwarp(act);
    }
}



// 仅当 npairs > 4 时使用 WMMA（两对一批：TL/BR），否则纯标量 5x5；
// sRhs 寄存器化；DinvL*rhs 标量 5x5（__shfl_sync）
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_tc_batch2_thresh(
    int nb,
    const int*  __restrict__ Abrow,   // 未用，保留接口一致
    const int*  __restrict__ Abcol,   // 未用
    const float*__restrict__ Abval,   // A 的块值 (float), BSR 顺序，每块 25 标量
    const int*  __restrict__ Mbrow,   // per-column rowptr
    const int*  __restrict__ Mcol,    // S_j
    const int*  __restrict__ PairPtr, // 对列表 rowptr（与 Mbrow/Mcol 对齐）
    const int*  __restrict__ PairPLocal,// p 的列内局部号（0..idx-1）
    const int*  __restrict__ PairSrc, // A 中 (t,p) 的 src_pos（Abcol 下标）
    float*      __restrict__ Mval,    // 输出（BCSC 顺序）
    const float*__restrict__ DinvL    // 预逆 (nb*25)
){
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    const int j = blockIdx.x * warpsPerBlock + warp;   // warp -> column
    if (j >= nb) return;

    const int start = Mbrow[j];
    const int end   = Mbrow[j+1];
    const int N     = end - start;
    if (N <= 0) return;

    // --- 每 warp 共享内存：2*BS2(小缓冲) + [128B 对齐] + 3*16*16(WMMA 区) ---
    extern __shared__ unsigned char smem[];
    constexpr int BYTES_SMALL = 2 * BS2 * sizeof(float);               // 200B
    constexpr int ALIGN       = 128;
    constexpr int PAD         = (ALIGN - (BYTES_SMALL % ALIGN)) % ALIGN; // 56B
    constexpr int BYTES_WMMA  = 3 * 16 * 16 * sizeof(float);           // 3072B
    unsigned char* warp_base  = smem + warp * (BYTES_SMALL + PAD + BYTES_WMMA);

    // 小缓冲（5x5）
    float* sA   = reinterpret_cast<float*>(warp_base + 0 * BS2 * sizeof(float));
    float* sYp  = reinterpret_cast<float*>(warp_base + 1 * BS2 * sizeof(float));

    // 16x16 WMMA 缓冲（128B 对齐）
    float* sA16 = reinterpret_cast<float*>(warp_base + BYTES_SMALL + PAD + 0 * 16 * 16 * sizeof(float));
    float* sB16 = reinterpret_cast<float*>(warp_base + BYTES_SMALL + PAD + 1 * 16 * 16 * sizeof(float));
    float* sC16 = reinterpret_cast<float*>(warp_base + BYTES_SMALL + PAD + 2 * 16 * 16 * sizeof(float));

    wmma::fragment<wmma::matrix_a, 16,16,8, wmma::precision::tf32, wmma::row_major> Af;
    wmma::fragment<wmma::matrix_b, 16,16,8, wmma::precision::tf32, wmma::row_major> Bf;
    wmma::fragment<wmma::accumulator,       16,16,8, float> Cf;
    // 每个 lane 对应的 (r,c)
    const int rr = lane / BS;
    const int cc = lane - rr * BS;

    // 活跃掩码（仅 0..24 参与）
    const unsigned FULL = 0xFFFFFFFFu;
    const unsigned act  = __ballot_sync(FULL, lane < BS2);

    // ---- 逐行块处理 ----
    for (int idx = 0; idx < N; ++idx) {
        const int t = Mcol[start + idx];

        // 本 lane 的 rhs 元素（寄存器化）
        float rhs_rc = (lane < BS2 && t == j && rr == cc) ? 1.f : 0.f;

        // 本行的 pair 区间与数量
        const int g0     = start + idx;
        const int qbeg   = PairPtr[g0];
        const int qend   = PairPtr[g0 + 1];
        const int npairs = qend - qbeg;

        // 累计 C = sum_k A*Yp（仅累加 TL(5x5) 元素到寄存器）
        float acc_rc = 0.f;

        if (npairs > 4) {
            // ---- 仅在需要 WMMA 时清一次 A16/B16（TR/BL 保持 0）----
            for (int t2 = lane; t2 < 16*16; t2 += 32) {
                sA16[t2] = 0.f;
                sB16[t2] = 0.f;
            }
            __syncwarp(act);

            // 两对一批（TL/BR），其余零头走标量
            int qb = qbeg;
            for (; qb + 1 < qend; qb += 2) {
                // 对0 -> TL(0,0)
                {
                    const int p0 = PairPLocal[qb + 0];
                    const int s0 = PairSrc   [qb + 0];
                    if (lane < BS2) {
                        sA [lane] = Abval[s0 * BS2 + lane];
                        sYp[lane] = Mval [(start + p0) * BS2 + lane];
                    }
                    __syncwarp(act);
                    for (int e = lane; e < BS2; e += 32) {
                        const int r = e / BS, c = e % BS;
                        sA16[(0 + r)*16 + (0 + c)] = sA [r*BS + c];
                        sB16[(0 + r)*16 + (0 + c)] = sYp[r*BS + c];
                    }
                    __syncwarp(act);
                }
                // 对1 -> BR(8,8)
                {
                    const int p1 = PairPLocal[qb + 1];
                    const int s1 = PairSrc   [qb + 1];
                    if (lane < BS2) {
                        sA [lane] = Abval[s1 * BS2 + lane];
                        sYp[lane] = Mval [(start + p1) * BS2 + lane];
                    }
                    __syncwarp(act);
                    for (int e = lane; e < BS2; e += 32) {
                        const int r = e / BS, c = e % BS;
                        sA16[(8 + r)*16 + (8 + c)] = sA [r*BS + c];
                        sB16[(8 + r)*16 + (8 + c)] = sYp[r*BS + c];
                    }
                    __syncwarp(act);
                }

                // 一次 WMMA：TL/BR 两块同时算
                wmma::fill_fragment(Cf, 0.0f);
                wmma::load_matrix_sync(Af, sA16, 16);
                wmma::load_matrix_sync(Bf, sB16, 16);
                wmma::mma_sync(Cf, Af, Bf, Cf);
                wmma::store_matrix_sync(sC16, Cf, 16, wmma::mem_row_major);
                __syncwarp(act);

                if (lane < BS2) {
                    acc_rc += sC16[rr*16 + cc];                   // TL
                    acc_rc += sC16[(8 + rr)*16 + (8 + cc)];       // BR
                }
                __syncwarp(act);
                // 下一批直接覆盖 TL/BR，无需再清 16×16
            }
            // 零头（奇数 1 对）→ 标量 5×5
            if (qb < qend) {
                const int p = PairPLocal[qb];
                const int s = PairSrc   [qb];
                if (lane < BS2) {
                    sA [lane] = Abval[s * BS2 + lane];
                    sYp[lane] = Mval [(start + p) * BS2 + lane];
                }
                __syncwarp(act);
                if (lane < BS2) {
                    float sum = 0.f;
                    #pragma unroll
                    for (int k = 0; k < BS; ++k)
                        sum += sA[rr*BS + k] * sYp[k*BS + cc];
                    acc_rc += sum;
                }
                __syncwarp(act);
            }
        } else {
            // ---- npairs <= 4：全部走标量 5×5 ----
            for (int q = qbeg; q < qend; ++q) {
                const int p = PairPLocal[q];
                const int s = PairSrc   [q];
                if (lane < BS2) {
                    sA [lane] = Abval[s * BS2 + lane];
                    sYp[lane] = Mval [(start + p) * BS2 + lane];
                }
                __syncwarp(act);
                if (lane < BS2) {
                    float sum = 0.f;
                    #pragma unroll
                    for (int k = 0; k < BS; ++k)
                        sum += sA[rr*BS + k] * sYp[k*BS + cc];
                    acc_rc += sum;
                }
                __syncwarp(act);
            }
        }

        // 一次性 rhs -= 累计的 C
        if (lane < BS2) rhs_rc -= acc_rc;
        __syncwarp(act);

        // ---- 标量 5×5：Y = DinvL[t] * rhs（__shfl_sync 聚合列）----
        if (lane < BS2) {
            const float* inv = DinvL + t * BS2;
            float y = 0.f;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                const int srcLane = k * BS + cc;       // 同列第 k 行
                const float rhs_kc = __shfl_sync(act, rhs_rc, srcLane);
                y += inv[rr*BS + k] * rhs_kc;
            }
            Mval[(start + idx) * BS2 + lane] = y;
        }
        __syncwarp(act);
    }
}
__global__ void count_npairs_gt4_kernel(int Mnnz,
                                        const int* __restrict__ PairPtr,
                                        unsigned int* __restrict__ out_count)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x; // g 对应 (j,idx) 的全局条目号
    if (g >= Mnnz) return;

    int npairs = PairPtr[g + 1] - PairPtr[g];      // 该条目的对数
    if (npairs > 4) {
        atomicAdd(out_count, 1u);
    }
}

// ==================== host helpers ====================

// 简单版：从 BSR(L 或 U) 构造 BCSC pattern：for_lower=true→L，false→U
void build_bcsc_pattern_from_bsr_scalarILU(
    int nb,
    const int* Abrow,
    const int* Abcol,
    bool for_lower,
    std::vector<int>& Mbrow,
    std::vector<int>& Mcol)
{
    std::vector<std::vector<int>> cols(nb);
    for (int br = 0; br < nb; ++br) {
        for (int p = Abrow[br]; p < Abrow[br + 1]; ++p) {
            int bc = Abcol[p];
            if (for_lower) {
                if (bc > br) continue; // keep lower(tri) blocks
            } else {
                if (bc < br) continue; // keep upper(tri) blocks
            }
            cols[bc].push_back(br);
        }
    }
    // make sure diagonal is present
    for (int j = 0; j < nb; ++j) cols[j].push_back(j);

    Mbrow.assign(nb + 1, 0);
    for (int j = 0; j < nb; ++j) {
        auto& v = cols[j];
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        Mbrow[j + 1] = Mbrow[j] + (int)v.size();
    }
    Mcol.resize(Mbrow[nb]);
    for (int j = 0; j < nb; ++j) {
        int off = Mbrow[j];
        auto& v = cols[j];
        for (int k = 0; k < (int)v.size(); ++k) {
            Mcol[off + k] = v[k];
        }
    }
}


//方法三L
__global__ void count_pairs_lower_kernel(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const int* __restrict__ Mbrow,   // per-column rowptr
    const int* __restrict__ Mcol,    // concatenated row indices (S_j)
    int* __restrict__ PairPtr        // out, length Mbrow[nb]+1; filled with counts here
){
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;

    int j = blockIdx.x * warpsPerBlock + warp; // warp->column
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j+1];
    int N     = end - start;
    if (N <= 0) return;

    // 为列 j 的每个 idx 计数
    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx]; // 行

        // 交集 Abcol[t,*] 与 S_j[0..N-1]，但只保留 p < idx（严格下三角）
        int a0 = Abrow[t], a1 = Abrow[t+1];
        int iA = (lane == 0) ? a0 : a1; // 仅 lane0 真用；其它 lane 占位
        int iS = 0;

        int cnt = 0;
        if (lane == 0) {
            while (iA < a1 && iS < N) {
                int colA = Abcol[iA];
                int colS = Mcol[start + iS];
                if (colA == colS) {
                    // 只统计 p_local < idx
                    if (iS < idx) ++cnt;
                    ++iA; ++iS;
                } else if (colA < colS) {
                    ++iA;
                } else {
                    ++iS;
                }
            }
            // 写到 PairPtr 的"计数位"
            int g = start + idx;      // 全局条目号
            PairPtr[g] = cnt;
        }
        __syncwarp();
    }

    // 列尾的 +1 位置置 0（为后续 exclusive_scan 做好末尾）
    if (lane == 0 && j == nb - 1) {
        PairPtr[ Mbrow[nb] ] = 0;
    }
}
__global__ void fill_pairs_lower_kernel(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const int* __restrict__ Mbrow,
    const int* __restrict__ Mcol,
    const int* __restrict__ PairPtrBase, // 扫描后的 PairPtr（前缀和）
    int* __restrict__ PairPLocal,        // out
    int* __restrict__ PairSrc            // out
){
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;

    int j = blockIdx.x * warpsPerBlock + warp;
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j+1];
    int N     = end - start;
    if (N <= 0) return;

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        int a0 = Abrow[t], a1 = Abrow[t+1];
        int iA = (lane == 0) ? a0 : a1;
        int iS = 0;

        if (lane == 0) {
            int out = PairPtrBase[start + idx];
            while (iA < a1 && iS < N) {
                int colA = Abcol[iA];
                int colS = Mcol[start + iS];
                if (colA == colS) {
                    if (iS < idx) {
                        // 命中一个 (t, p=S_j[iS])
                        PairPLocal[out] = iS;      // 局部列号（p_local）
                        PairSrc  [out] = iA;       // A 的源位置
                        ++out;
                    }
                    ++iA; ++iS;
                } else if (colA < colS) {
                    ++iA;
                } else {
                    ++iS;
                }
            }
        }
        __syncwarp();
    }
}
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    const int* __restrict__ Mbrow,     // per-column rowptr
    const int* __restrict__ Mcol,      // per-column rows (S_j)
    // 新增：对列表（与 Mbrow/Mcol 对齐）
    const int* __restrict__ PairPtr,   // length Mbrow[nb]+1
    const int* __restrict__ PairPLocal,// length PairPtr[end]
    const int* __restrict__ PairSrc,   // length PairPtr[end]
    // 输出
    VALUE_TYPE* __restrict__ Mval,         // BCSC blocks
    const VALUE_TYPE* __restrict__ DinvL   // nb*BS2
){
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;

    int j = blockIdx.x * warpsPerBlock + warp; // warp->column
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N <= 0) return;

    extern __shared__ VALUE_TYPE sh[];
    const int STRIDE = 3 * BS2; // rhs + Ablk + Yp
    VALUE_TYPE* base = sh + warp * STRIDE;
    VALUE_TYPE* sRhs = base + 0 * BS2;
    VALUE_TYPE* sA   = base + 1 * BS2;
    VALUE_TYPE* sYp  = base + 2 * BS2;

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        // rhs = (t==j ? I : 0)
        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? 1.0 : 0.0;
        }
        __syncwarp();

        // 遍历对列表：所有与本行 t 相关、且 p_local < idx 的 (t, p)
        int g0 = start + idx;
        int qbeg = PairPtr[g0];
        int qend = PairPtr[g0 + 1];
        VALUE_TYPE sum = 0.0;
        for (int q = qbeg; q < qend; ++q) {
            int p_local = PairPLocal[q];        // 0..idx-1
            int srcA    = PairSrc[q];           // A 的 BSR 位置

            // 读 A(t, p) -> sA （warp 协作）
            if (lane < BS2) {
                sA[lane] = Abval[srcA * BS2 + lane];
                // 读 Y_p -> sYp
                sYp[lane] = Mval[(start + p_local) * BS2 + lane];
            }
            __syncwarp();

            // sRhs -= sA * sYp
            if (lane < BS2) {
                int rr = lane / BS;
                int cc = lane - rr * BS;
                #pragma unroll
                for (int kk = 0; kk < BS; ++kk) {
                    sum += sA[rr * BS + kk] * sYp[kk * BS + cc];
                }
            }
            __syncwarp();
        }
        sRhs[lane] -= sum;

        // Y = DinvL[t] * rhs
        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            int rr = lane / BS;
            int cc = lane - rr * BS;
            VALUE_TYPE acc = 0.0;
            #pragma unroll
            for (int kk = 0; kk < BS; ++kk) {
                acc += inv[rr * BS + kk] * sRhs[kk * BS + cc];
            }
            Mval[(start + idx) * BS2 + lane] = acc;
        }
        __syncwarp();
    }
}
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_noshm(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    const int* __restrict__ Mbrow,       // per-column rowptr
    const int* __restrict__ Mcol,        // per-column rows (S_j)
    const int* __restrict__ PairPtr,     // length Mbrow[nb]+1
    const int* __restrict__ PairPLocal,  // length PairPtr[end]
    const int* __restrict__ PairSrc,     // length PairPtr[end]
    VALUE_TYPE* __restrict__ Mval,       // BCSC blocks
    const VALUE_TYPE* __restrict__ DinvL // nb*BS2
){
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    const int j = blockIdx.x * warpsPerBlock + warp; // warp -> column
    if (j >= nb) return;

    const int start = Mbrow[j];
    const int end   = Mbrow[j + 1];
    const int N     = end - start;
    if (N <= 0) return;

    // 只让前 BS2 个 lane 参与真实计算
    const unsigned mask = __ballot_sync(0xffffffff, lane < BS2);

    for (int idx = 0; idx < N; ++idx) {
        const int g0 = start + idx;
        const int t  = Mcol[g0];

        VALUE_TYPE rhs_val = (VALUE_TYPE)0;

        int rr = 0, cc = 0;
        if (lane < BS2) {
            rr = lane / BS;
            cc = lane % BS;
            rhs_val = (t == j && rr == cc) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }

        // 遍历所有 (t, p)，其中 p_local < idx
        const int qbeg = PairPtr[g0];
        const int qend = PairPtr[g0 + 1];

        if (lane < BS2) {
            for (int q = qbeg; q < qend; ++q) {
                const int p_local = PairPLocal[q];
                const int srcA    = PairSrc[q];

                VALUE_TYPE sum = (VALUE_TYPE)0;

                #pragma unroll
                for (int kk = 0; kk < BS; ++kk) {
                    const VALUE_TYPE a =
                        Abval[srcA * BS2 + rr * BS + kk];
                    const VALUE_TYPE y =
                        Mval[(start + p_local) * BS2 + kk * BS + cc];
                    sum += a * y;
                }

                rhs_val -= sum;
            }
        }

        // Y = DinvL[t] * rhs
        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            VALUE_TYPE acc = (VALUE_TYPE)0;

            #pragma unroll
            for (int kk = 0; kk < BS; ++kk) {
                // rhs[kk, cc] 位于 lane = kk*BS + cc
                const int src_lane = kk * BS + cc;
                const VALUE_TYPE rhs_kc = __shfl_sync(mask, rhs_val, src_lane);
                acc += inv[rr * BS + kk] * rhs_kc;
            }

            Mval[(start + idx) * BS2 + lane] = acc;
        }
    }
}

// 直接使用shared memory 传递warp内依赖
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY(
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
    int Nmax_perwarp              // 传入：每 warp 预留的最大列长
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

    // 防呆：如果某列 N > Nmax_perwarp，就直接返回或改成走 global（看你策略）
    if (N > Nmax_perwarp) return;

    // shared per-warp:
    //  sRhs(25) + sA(25) + Ycache(Nmax*25)
    extern __shared__ VALUE_TYPE sh[];
    VALUE_TYPE* warp_base = sh + warp * ( (2*BS2) + Nmax_perwarp*BS2 );

    VALUE_TYPE* sRhs   = warp_base + 0;
    VALUE_TYPE* sA     = warp_base + BS2;
    VALUE_TYPE* sYall  = warp_base + 2*BS2;                 // [Nmax][25]

    // 可选：把缓存区清零（不是必须；我们会覆盖写每个 idx 的 25 元素）
    // if (lane < Nmax_perwarp*BS2) sYall[lane] = 0;
    // __syncwarp();

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        // 用 Pair 列表累减：rhs -= sum A(t,p)*Yp
        int g0   = start + idx;
        int qbeg = PairPtr[g0];
        int qend = PairPtr[g0 + 1];
        VALUE_TYPE* Ycur = sYall + idx * BS2;

        // rhs = (t==j ? I : 0)
        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }
        __syncwarp();

        for (int q = qbeg; q < qend; ++q) {
            int p_local = PairPLocal[q];  // 0..idx-1
            int srcA    = PairSrc[q];

            // 读 A(t,p) -> sA
            if (lane < BS2) {
                sA[lane] = Abval[srcA * BS2 + lane];
            }
            // 读 Yp：从 shared 缓存读，不再从 global 读
            if (lane < BS2) {
                // p_local 一定 < idx，因此一定已经写入 sYall
                // sYall 的布局：连续存 idx=0..N-1 的块
                // 每块 25 个
                // 取出第 p_local 块
                const VALUE_TYPE* Yp = sYall + p_local * BS2;
                // 为了保持你原逻辑，我们用 sRhs -= sA*Yp 直接算，不必再拷到 sYp 临时数组
                int rr = lane / BS;
                int cc = lane - rr * BS;
                VALUE_TYPE sum = 0;
                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    sum += sA[rr * BS + k] * Yp[k * BS + cc];
                }
                sRhs[lane] -= sum;
            }
            __syncwarp();
        }

        // Y = DinvL[t] * rhs（标量 5×5）
        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            int rr = lane / BS;
            int cc = lane - rr * BS;
            VALUE_TYPE acc = 0;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                acc += inv[rr * BS + k] * sRhs[k * BS + cc];
            }
            // 先写到 shared 缓存（本列内后续会用到）
            sYall[idx * BS2 + lane] = acc;
        }
        __syncwarp();
    }

    // 最后一次性把本列所有块写回 global（减少 global 往返）
    for (int idx = 0; idx < N; ++idx) {
        if (lane < BS2) {
            Mval[(start + idx) * BS2 + lane] = sYall[idx * BS2 + lane];
        }
        __syncwarp();
    }
}

// A_embed: 8x4
// 从 sA(5x5,row-major) 里取前4列，组成 5x4；再补到 8x4
__device__ __forceinline__
double load_fragA_from_sA_5x5_first4cols(const double* sA, int lane)
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
double load_fragB_from_Yp_5x5_first4rows(const double* Yp, int lane)
{
    int row = lane &  3;   // 0..3
    int col = lane >> 2;   // 0..7

    if (col < 5) {
        return Yp[row * 5 + col];
    } else {
        return 0.0;
    }
}

// Acc/C/D 对应 8x8
// 只把左上 5x5 映射到 sRhs，其余补0
__device__ __forceinline__
void load_acc_from_sRhs_5x5(const double* sRhs, int lane, double acc[2])
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
void store_acc_to_sRhs_5x5(double* sRhs, int lane, const double acc[2])
{
    int groupID           = lane >> 2;
    int threadID_in_group = lane &  3;

    int row = groupID;
    int c0  = threadID_in_group * 2 + 0;
    int c1  = threadID_in_group * 2 + 1;

    if (row < 5 && c0 < 5) sRhs[row * 5 + c0] = acc[0];
    if (row < 5 && c1 < 5) sRhs[row * 5 + c1] = acc[1];
}

// A_embed: 8x4
// 从 inv(5x5,row-major) 里取前4列，组成 5x4；再补到 8x4
__device__ __forceinline__
double load_fragA_from_inv_5x5_first4cols(const double* inv, int lane)
{
    int row = lane >> 2;   // 0..7
    int col = lane &  3;   // 0..3

    if (row < 5) return inv[row * 5 + col];
    return 0.0;
}

// B_embed: 4x8
// 从 rhs(5x5,row-major) 里取前4行，组成 4x5；再补到 4x8
__device__ __forceinline__
double load_fragB_from_rhs_5x5_first4rows(const double* rhs, int lane)
{
    int row = lane &  3;   // 0..3
    int col = lane >> 2;   // 0..7

    if (col < 5) return rhs[row * 5 + col];
    return 0.0;
}

// Acc 初始化为 0，对应 8x8 accumulator
__device__ __forceinline__
void load_acc_zero_5x5(int lane, double acc[2])
{
    acc[0] = 0.0;
    acc[1] = 0.0;
}

// 只把左上 5x5 写回 Y
__device__ __forceinline__
void store_acc_to_Y_5x5(double* Y, int lane, const double acc[2])
{
    int row = lane >> 2;       // 0..7
    int tid = lane & 3;        // 0..3

    int c0 = tid * 2 + 0;
    int c1 = tid * 2 + 1;

    if (row < 5 && c0 < 5) Y[row * 5 + c0] = acc[0];
    if (row < 5 && c1 < 5) Y[row * 5 + c1] = acc[1];
}

// 做：Y = inv[:,0:4] * rhs[0:4,:]
// 注意：只算 k=0..3，k=4 的 tail 后面单独补
__device__ __forceinline__
void tc_gemm_Dinv_rhs_5x5_k4_store(
    double* Y,
    const double* inv,
    const double* rhs
){
#if __CUDA_ARCH__ >= 800
    int lane = threadIdx.x & 31;

    double fragA = load_fragA_from_inv_5x5_first4cols(inv, lane);
    double fragB = load_fragB_from_rhs_5x5_first4rows(rhs, lane);

    double acc[2];
    load_acc_zero_5x5(lane, acc);

    // acc = inv[:,0:4] * rhs[0:4,:]
    mma_m8n8k4_f64(acc, fragA, fragB);

    store_acc_to_Y_5x5(Y, lane, acc);
#endif
}

// 补 k=4：Y += inv[:,4] * rhs[4,:]
__device__ __forceinline__
void add_tail_Dinv_rhs_k4_outer_product(
    double* Y,
    const double* inv,
    const double* rhs
){
    int lane = threadIdx.x & 31;

    if (lane < 25) {
        int rr = lane / 5;
        int cc = lane % 5;
        Y[rr * 5 + cc] += inv[rr * 5 + 4] * rhs[4 * 5 + cc];
    }
}

__device__ __forceinline__
double load_double_cs_bsr5(const double* p)
{
    double v;
#if defined(__CUDA_ARCH__)
    asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(v) : "l"(p));
#else
    v = *p;
#endif
    return v;
}

__device__ __forceinline__
double load_fragA_from_global_A_5x5_first4cols(const double* Ablk, int lane)
{
    int row = lane >> 2;
    int col = lane & 3;
    return (row < 5) ? load_double_cs_bsr5(Ablk + row * 5 + col) : 0.0;
}

// 做：sRhs -= A_use(5x4) * Yp_use(4x5)
// 注意：必须整 warp 调用
__device__ __forceinline__
void tc_update_rhs_5x5_k4(double* sRhs, const double* sA, const double* Yp)
{
#if __CUDA_ARCH__ >= 800
    int lane = threadIdx.x & 31;

    double fragA = load_fragA_from_sA_5x5_first4cols(sA, lane);
    double fragB = load_fragB_from_Yp_5x5_first4rows(Yp, lane);
    //double fragA=1;
    //double fragB=1;
    double acc[2];

    load_acc_from_sRhs_5x5(sRhs, lane, acc);
    // sRhs += (-A) * B
    fragA = -fragA;

    mma_m8n8k4_f64(acc, fragA, fragB);

    store_acc_to_sRhs_5x5(sRhs, lane, acc);
#endif
}

__device__ __forceinline__
void tc_update_rhs_5x5_k4_Aglobal_Yshared(
    double* sRhs,
    const double* Ablk,
    const double* Yp)
{
#if __CUDA_ARCH__ >= 800
    int lane = threadIdx.x & 31;

    double fragA = load_fragA_from_global_A_5x5_first4cols(Ablk, lane);
    double fragB = load_fragB_from_Yp_5x5_first4rows(Yp, lane);
    double acc[2];

    load_acc_from_sRhs_5x5(sRhs, lane, acc);
    fragA = -fragA;

    mma_m8n8k4_f64(acc, fragA, fragB);

    store_acc_to_sRhs_5x5(sRhs, lane, acc);
#endif
}

__device__ __forceinline__
void add_tail_k4_outer_product(double* sRhs, const double* sA, const double* Yp)
{
    int lane = threadIdx.x & 31;

    if (lane < 25) {
        int rr = lane / 5;
        int cc = lane % 5;

        //double a = sA[rr * 5 + 4];   // A 的第 5 列
        //double b = Yp[4 * 5 + cc];   // B 的第 5 行
        sRhs[rr * 5 + cc] -= sA[rr * 5 + 4] * Yp[4 * 5 + cc];
    }
}

__device__ __forceinline__
void add_tail_k4_outer_product_Aglobal_Yshared(
    double* sRhs,
    const double* Ablk,
    const double* Yp)
{
    int lane = threadIdx.x & 31;

    if (lane < 25) {
        int rr = lane / 5;
        int cc = lane % 5;
        double a = load_double_cs_bsr5(Ablk + rr * 5 + 4);
        sRhs[rr * 5 + cc] -= a * Yp[4 * 5 + cc];
    }
}


__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY_tensor(
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
    int Nmax_perwarp              // 传入：每 warp 预留的最大列长
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

    // 防呆：如果某列 N > Nmax_perwarp，就直接返回或改成走 global（看你策略）
    if (N > Nmax_perwarp) return;

    // shared per-warp:
    //  sRhs(25) + sA(25) + Ycache(Nmax*25)
    extern __shared__ VALUE_TYPE sh[];
    VALUE_TYPE* warp_base = sh + warp * ((2 * BS2) + Nmax_perwarp * BS2);

    VALUE_TYPE* sRhs   = warp_base + 0;
    VALUE_TYPE* sA     = warp_base + BS2;
    VALUE_TYPE* sYall  = warp_base + 2 * BS2;                 // [Nmax][25]

    // 可选：把缓存区清零（不是必须；我们会覆盖写每个 idx 的 25 元素）
    // if (lane < Nmax_perwarp*BS2) sYall[lane] = 0;
    // __syncwarp();

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        // 用 Pair 列表累减：rhs -= sum A(t,p)*Yp
        int g0   = start + idx;
        int qbeg = PairPtr[g0];
        int qend = PairPtr[g0 + 1];
        VALUE_TYPE* Ycur = sYall + idx * BS2;

        // rhs = (t==j ? I : 0)
        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }
        __syncwarp();

        for (int q = qbeg; q < qend; ++q) {
            int p_local = PairPLocal[q];  // 0..idx-1
            int srcA    = PairSrc[q];

            if (lane < BS2) {
                sA[lane] = Abval[srcA * BS2 + lane];
            }
            {
                const VALUE_TYPE* Yp = sYall + p_local * BS2;

            #if __CUDA_ARCH__ >= 800
                tc_update_rhs_5x5_k4((double*)sRhs, (const double*)sA, (const double*)Yp);

                add_tail_k4_outer_product(sRhs, sA, Yp);
            #else
                // fallback: 原始标量版本
                if (lane < BS2) {
                    int rr = lane / BS;
                    int cc = lane - rr * BS;
                    VALUE_TYPE sum = 0;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {   // 这里只算前4列/前4行
                        sum += sA[rr * BS + k] * Yp[k * BS + cc];
                    }
                    sRhs[lane] -= sum;
                }
            #endif
               
            }
            __syncwarp();

        }

          {
            const VALUE_TYPE* inv  = DinvL + t * BS2;

        #if __CUDA_ARCH__ >= 800
            tc_gemm_Dinv_rhs_5x5_k4_store(
                (double*)Ycur,
                (const double*)inv,
                (const double*)sRhs
            );

            __syncwarp();

            add_tail_Dinv_rhs_k4_outer_product(
                (double*)Ycur,
                (const double*)inv,
                (const double*)sRhs
            );
        #else
            if (lane < BS2) {
                int rr = lane / BS;
                int cc = lane - rr * BS;
                VALUE_TYPE acc = 0;

                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    acc += inv[rr * BS + k] * sRhs[k * BS + cc];
                }

                Ycur[lane] = acc;
            }
        #endif
        }
        __syncwarp();
    }

    for (int idx = 0; idx < N; ++idx) {
        if (lane < BS2) {
            Mval[(start + idx) * BS2 + lane] = sYall[idx * BS2 + lane];
        }
        __syncwarp();
    }
}

__device__ __forceinline__
void isai_lower_bsr5_light_globalY_tensor_col_body(
    int j,
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
    VALUE_TYPE* sh,
    int warp)
{
    int lane = threadIdx.x & 31;
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N <= 0) return;

    VALUE_TYPE* warp_base = sh + warp * (2 * BS2);
    VALUE_TYPE* sRhs = warp_base + 0;
    VALUE_TYPE* sA   = warp_base + BS2;

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
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

#if __CUDA_ARCH__ >= 800
            tc_update_rhs_5x5_k4((double*)sRhs, (const double*)sA, (const double*)Yp);
            add_tail_k4_outer_product(sRhs, sA, Yp);
#else
            if (lane < BS2) {
                int rr = lane / BS;
                int cc = lane - rr * BS;
                VALUE_TYPE sum = 0;
                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    sum += sA[rr * BS + k] * Yp[k * BS + cc];
                }
                sRhs[lane] -= sum;
            }
#endif
            __syncwarp();
        }

        VALUE_TYPE* Ycur = Mval + (start + idx) * BS2;
        const VALUE_TYPE* inv = DinvL + t * BS2;

#if __CUDA_ARCH__ >= 800
        tc_gemm_Dinv_rhs_5x5_k4_store(
            (double*)Ycur,
            (const double*)inv,
            (const double*)sRhs
        );
        __syncwarp();
        add_tail_Dinv_rhs_k4_outer_product(
            (double*)Ycur,
            (const double*)inv,
            (const double*)sRhs
        );
#else
        if (lane < BS2) {
            int rr = lane / BS;
            int cc = lane - rr * BS;
            VALUE_TYPE acc = 0;

            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                acc += inv[rr * BS + k] * sRhs[k * BS + cc];
            }

            Ycur[lane] = acc;
        }
#endif
        __syncwarp();
    }
}

__device__ __forceinline__
void isai_lower_bsr5_light_globalY_scalar_col_body(
    int j,
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
    const VALUE_TYPE* __restrict__ DinvL)
{
    int lane = threadIdx.x & 31;
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N <= 0) return;

    unsigned active = __ballot_sync(0xffffffff, lane < BS2);

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];
        VALUE_TYPE rhs = 0;
        int rr = 0;
        int cc = 0;

        if (lane < BS2) {
            rr = lane / BS;
            cc = lane - rr * BS;
            rhs = (t == j && rr == cc) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }

        int g0   = start + idx;
        int qbeg = PairPtr[g0];
        int qend = PairPtr[g0 + 1];

        for (int q = qbeg; q < qend; ++q) {
            int p_local = PairPLocal[q];
            int srcA    = PairSrc[q];

            if (lane < BS2) {
                const VALUE_TYPE* Ablk = Abval + srcA * BS2;
                const VALUE_TYPE* Yp   = Mval + (start + p_local) * BS2;

                VALUE_TYPE sum = 0;
                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    sum += Ablk[rr * BS + k] * Yp[k * BS + cc];
                }
                rhs -= sum;
            }
        }

        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            VALUE_TYPE y = 0;

            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                VALUE_TYPE rhs_k = __shfl_sync(active, rhs, k * BS + cc);
                y += inv[rr * BS + k] * rhs_k;
            }

            Mval[(start + idx) * BS2 + lane] = y;
        }
        __syncwarp();
    }
}

__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_light_globalY_scalar_cols(
    int nb,
    const int* __restrict__ WorkCols,
    int nWorkCols,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    const int* __restrict__ Mbrow,
    const int* __restrict__ Mcol,
    const int* __restrict__ PairPtr,
    const int* __restrict__ PairPLocal,
    const int* __restrict__ PairSrc,
    VALUE_TYPE* __restrict__ Mval,
    const VALUE_TYPE* __restrict__ DinvL)
{
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;
    int work = blockIdx.x * warpsPerBlock + warp;
    if (work >= nWorkCols) return;

    int j = WorkCols[work];
    isai_lower_bsr5_light_globalY_scalar_col_body(
        j, nb, Abrow, Abcol, Abval, Mbrow, Mcol,
        PairPtr, PairPLocal, PairSrc, Mval, DinvL);
}

__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_light_globalY_tensor_cols(
    int nb,
    const int* __restrict__ WorkCols,
    int nWorkCols,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    const int* __restrict__ Mbrow,
    const int* __restrict__ Mcol,
    const int* __restrict__ PairPtr,
    const int* __restrict__ PairPLocal,
    const int* __restrict__ PairSrc,
    VALUE_TYPE* __restrict__ Mval,
    const VALUE_TYPE* __restrict__ DinvL)
{
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;
    int work = blockIdx.x * warpsPerBlock + warp;
    if (work >= nWorkCols) return;

    extern __shared__ VALUE_TYPE sh[];
    int j = WorkCols[work];
    isai_lower_bsr5_light_globalY_tensor_col_body(
        j, nb, Abrow, Abcol, Abval, Mbrow, Mcol,
        PairPtr, PairPLocal, PairSrc, Mval, DinvL, sh, warp);
}

__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_globalY_tensor(
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
    const VALUE_TYPE* __restrict__ DinvL)
{
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;
    int j = blockIdx.x * warpsPerBlock + warp;

    extern __shared__ VALUE_TYPE sh[];
    isai_lower_bsr5_light_globalY_tensor_col_body(
        j, nb, Abrow, Abcol, Abval, Mbrow, Mcol,
        PairPtr, PairPLocal, PairSrc, Mval, DinvL, sh, warp);
}

__device__ __forceinline__
void isai_lower_bsr5_cachedY_tensor_col_body(
    int j,
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
    int Nmax_perwarp,
    VALUE_TYPE* sh,
    int warp)
{
    int lane = threadIdx.x & 31;
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N <= 0 || N > Nmax_perwarp) return;

    VALUE_TYPE* warp_base = sh + warp * ((2 * BS2) + Nmax_perwarp * BS2);
    VALUE_TYPE* sRhs  = warp_base + 0;
    VALUE_TYPE* sA    = warp_base + BS2;
    VALUE_TYPE* sYall = warp_base + 2 * BS2;

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
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

            const VALUE_TYPE* Yp = sYall + p_local * BS2;

#if __CUDA_ARCH__ >= 800
            tc_update_rhs_5x5_k4((double*)sRhs, (const double*)sA, (const double*)Yp);
            add_tail_k4_outer_product(sRhs, sA, Yp);
#else
            if (lane < BS2) {
                int rr = lane / BS;
                int cc = lane - rr * BS;
                VALUE_TYPE sum = 0;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    sum += sA[rr * BS + k] * Yp[k * BS + cc];
                }
                sRhs[lane] -= sum;
            }
#endif
            __syncwarp();
        }

        const VALUE_TYPE* inv  = DinvL + t * BS2;
        VALUE_TYPE*       Ycur = sYall + idx * BS2;

#if __CUDA_ARCH__ >= 800
        tc_gemm_Dinv_rhs_5x5_k4_store(
            (double*)Ycur,
            (const double*)inv,
            (const double*)sRhs
        );
        __syncwarp();
        add_tail_Dinv_rhs_k4_outer_product(
            (double*)Ycur,
            (const double*)inv,
            (const double*)sRhs
        );
#else
        if (lane < BS2) {
            int rr = lane / BS;
            int cc = lane - rr * BS;
            VALUE_TYPE acc = 0;

            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                acc += inv[rr * BS + k] * sRhs[k * BS + cc];
            }

            Ycur[lane] = acc;
        }
#endif
        __syncwarp();
    }

    for (int idx = 0; idx < N; ++idx) {
        if (lane < BS2) {
            Mval[(start + idx) * BS2 + lane] = sYall[idx * BS2 + lane];
        }
        __syncwarp();
    }
}

__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY_tensor_shortcols(
    int nb,
    const int* __restrict__ WorkCols,
    int nWorkCols,
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
    int Nmax_perwarp)
{
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;
    int work = blockIdx.x * warpsPerBlock + warp;
    if (work >= nWorkCols) return;

    extern __shared__ VALUE_TYPE sh[];
    int j = WorkCols[work];
    isai_lower_bsr5_cachedY_tensor_col_body(
        j, nb, Abrow, Abcol, Abval, Mbrow, Mcol,
        PairPtr, PairPLocal, PairSrc, Mval, DinvL,
        Nmax_perwarp, sh, warp);
}

__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY_tensor_longcols(
    int nb,
    const int* __restrict__ WorkCols,
    int nWorkCols,
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
    int Nmax_perwarp)
{
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;
    int work = blockIdx.x * warpsPerBlock + warp;
    if (work >= nWorkCols) return;

    extern __shared__ VALUE_TYPE sh[];
    int j = WorkCols[work];
    isai_lower_bsr5_cachedY_tensor_col_body(
        j, nb, Abrow, Abcol, Abval, Mbrow, Mcol,
        PairPtr, PairPLocal, PairSrc, Mval, DinvL,
        Nmax_perwarp, sh, warp);
}
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY_tensor_cacheN(
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
    const VALUE_TYPE* __restrict__ DinvL
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

    // shared per-warp:
    // sRhs(25) + sA(25) + Ycache(CACHE_N * 25)
    extern __shared__ VALUE_TYPE sh[];

    VALUE_TYPE* warp_base =
        sh + warp * ((2 * BS2) + CACHE_N * BS2);

    VALUE_TYPE* sRhs  = warp_base;
    VALUE_TYPE* sA    = warp_base + BS2;
    VALUE_TYPE* sYall = warp_base + 2 * BS2;   // [CACHE_N][25]

    VALUE_TYPE* Mval_col_base = Mval + start * BS2;

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        // rhs = (t == j ? I : 0)
        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c)
                       ? (VALUE_TYPE)1
                       : (VALUE_TYPE)0;
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

            const VALUE_TYPE* Yp =
                (p_local < CACHE_N)
                ? (sYall + p_local * BS2)
                : (Mval_col_base + p_local * BS2);

#if __CUDA_ARCH__ >= 800
            tc_update_rhs_5x5_k4(
                (double*)sRhs,
                (const double*)sA,
                (const double*)Yp
            );

            add_tail_k4_outer_product(
                (double*)sRhs,
                (const double*)sA,
                (const double*)Yp
            );
#else
            if (lane < BS2) {
                int rr = lane / BS;
                int cc = lane - rr * BS;

                VALUE_TYPE sum = 0;

                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    sum += sA[rr * BS + k] * Yp[k * BS + cc];
                }

                sRhs[lane] -= sum;
            }
#endif

            __syncwarp();
        }

        const VALUE_TYPE* inv = DinvL + t * BS2;

        VALUE_TYPE* Ycur =
            (idx < CACHE_N)
            ? (sYall + idx * BS2)
            : (Mval_col_base + idx * BS2);

#if __CUDA_ARCH__ >= 800
        tc_gemm_Dinv_rhs_5x5_k4_store(
            (double*)Ycur,
            (const double*)inv,
            (const double*)sRhs
        );

        __syncwarp();

        add_tail_Dinv_rhs_k4_outer_product(
            (double*)Ycur,
            (const double*)inv,
            (const double*)sRhs
        );
#else
        if (lane < BS2) {
            int rr = lane / BS;
            int cc = lane - rr * BS;

            VALUE_TYPE acc = 0;

            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                acc += inv[rr * BS + k] * sRhs[k * BS + cc];
            }

            Ycur[lane] = acc;
        }
#endif

        __syncwarp();
    }

    // 只写回 shared cache 中的部分。
    // idx >= CACHE_N 的部分已经在计算时直接写入 Mval。
    int ncache = (N < CACHE_N) ? N : CACHE_N;

    for (int idx = 0; idx < ncache; ++idx) {
        if (lane < BS2) {
            Mval_col_base[idx * BS2 + lane] =
                sYall[idx * BS2 + lane];
        }

        __syncwarp();
    }
}
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY_tensor_cpasyncA(
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
    if (N > Nmax_perwarp) return;

    extern __shared__ VALUE_TYPE sh[];

    // per warp layout:
    // sRhs(25) + sA0(25) + sA1(25) + sYall(Nmax*25)
    VALUE_TYPE* warp_base = sh + warp * (3 * BS2 + Nmax_perwarp * BS2);

    VALUE_TYPE* sRhs   = warp_base + 0 * BS2;
    VALUE_TYPE* sA0    = warp_base + 1 * BS2;
    VALUE_TYPE* sA1    = warp_base + 2 * BS2;
    VALUE_TYPE* sYall  = warp_base + 3 * BS2;   // [Nmax_perwarp][25]

    VALUE_TYPE* sAbuf[2] = {sA0, sA1};

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        // rhs = (t == j ? I : 0)
        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }
        __syncwarp();

        int g0   = start + idx;
        int qbeg = PairPtr[g0];
        int qend = PairPtr[g0 + 1];

#if __CUDA_ARCH__ >= 800
        if (qbeg < qend) {
            int cur = 0;

            // 先预取第一个 A block -> sA0
            {
                int srcA0 = PairSrc[qbeg];
                const VALUE_TYPE* gA0 = Abval + srcA0 * BS2;

                prefetch_block25_A_cp_async(sAbuf[cur], gA0, lane);
                cp_async_commit();
                cp_async_wait_group<0>();
                __syncwarp();
            }

            for (int q = qbeg; q < qend; ++q) {
                int nxt = cur ^ 1;

                // 预取下一块 A 到另一个 buffer
                if (q + 1 < qend) {
                    int srcA_next = PairSrc[q + 1];
                    const VALUE_TYPE* gA_next = Abval + srcA_next * BS2;

                    prefetch_block25_A_cp_async(sAbuf[nxt], gA_next, lane);
                    cp_async_commit();
                }

                int p_local = PairPLocal[q];
                const VALUE_TYPE* Yp = sYall + p_local * BS2;

                tc_update_rhs_5x5_k4(
                    (double*)sRhs,
                    (const double*)sAbuf[cur],
                    (const double*)Yp
                );
                __syncwarp();

                add_tail_k4_outer_product(
                    sRhs,
                    sAbuf[cur],
                    Yp
                );
                __syncwarp();

                if (q + 1 < qend) {
                    cp_async_wait_group<0>();
                    __syncwarp();
                    cur = nxt;
                }
            }
        }
#else
        for (int q = qbeg; q < qend; ++q) {
            int p_local = PairPLocal[q];
            int srcA    = PairSrc[q];

            if (lane < BS2) {
                sA0[lane] = Abval[srcA * BS2 + lane];
            }
            __syncwarp();

            {
                const VALUE_TYPE* Yp = sYall + p_local * BS2;

                if (lane < BS2) {
                    int rr = lane / BS;
                    int cc = lane - rr * BS;
                    VALUE_TYPE sum = 0;
                    #pragma unroll
                    for (int k = 0; k < BS; ++k) {
                        sum += sA0[rr * BS + k] * Yp[k * BS + cc];
                    }
                    sRhs[lane] -= sum;
                }
            }
            __syncwarp();
        }
#endif

        // Y = DinvL[t] * rhs
        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            int rr = lane / BS;
            int cc = lane - rr * BS;
            VALUE_TYPE acc = 0;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                acc += inv[rr * BS + k] * sRhs[k * BS + cc];
            }
            sYall[idx * BS2 + lane] = acc;
        }
        __syncwarp();
    }

    // 最后统一写回 global
    for (int idx = 0; idx < N; ++idx) {
        if (lane < BS2) {
            Mval[(start + idx) * BS2 + lane] = sYall[idx * BS2 + lane];
        }
        __syncwarp();
    }
}

__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_tensor(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const VALUE_TYPE* __restrict__ Abval,
    const int* __restrict__ Mbrow,     // per-column rowptr
    const int* __restrict__ Mcol,      // per-column rows (S_j)
    // 新增：对列表（与 Mbrow/Mcol 对齐）
    const int* __restrict__ PairPtr,   // length Mbrow[nb]+1
    const int* __restrict__ PairPLocal,// length PairPtr[end]
    const int* __restrict__ PairSrc,   // length PairPtr[end]
    // 输出
    VALUE_TYPE* __restrict__ Mval,         // BCSC blocks
    const VALUE_TYPE* __restrict__ DinvL   // nb*BS2
){
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;

    int j = blockIdx.x * warpsPerBlock + warp; // warp->column
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N <= 0) return;

    extern __shared__ VALUE_TYPE sh[];
    const int STRIDE = 3 * BS2; // rhs + Ablk + Yp
    VALUE_TYPE* base = sh + warp * STRIDE;
    VALUE_TYPE* sRhs = base + 0 * BS2;
    VALUE_TYPE* sA   = base + 1 * BS2;
    VALUE_TYPE* sYp  = base + 2 * BS2;

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        // rhs = (t==j ? I : 0)
        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? 1.0 : 0.0;
        }
        __syncwarp();

        // 遍历对列表：所有与本行 t 相关、且 p_local < idx 的 (t, p)
        int g0 = start + idx;
        int qbeg = PairPtr[g0];
        int qend = PairPtr[g0 + 1];

        for (int q = qbeg; q < qend; ++q) {
            int p_local = PairPLocal[q];        // 0..idx-1
            int srcA    = PairSrc[q];           // A 的 BSR 位置

            // 读 A(t, p) -> sA （warp 协作）
            if (lane < BS2) {
                sA[lane] = Abval[srcA * BS2 + lane];
                // 读 Y_p -> sYp
                sYp[lane] = Mval[(start + p_local) * BS2 + lane];
            }
            __syncwarp();
            #if __CUDA_ARCH__ >= 800
                tc_update_rhs_5x5_k4((double*)sRhs, (const double*)sA, (const double*)sYp);
                __syncwarp();

                add_tail_k4_outer_product(sRhs, sA, sYp);
                
            #else
                // fallback: 原始标量版本
                if (lane < BS2) {
                    int rr = lane / BS;
                    int cc = lane - rr * BS;
                    VALUE_TYPE sum = 0.0;
                    #pragma unroll
                    for (int kk = 0; kk < BS; ++kk) {
                        sum += sA[rr * BS + kk] * sYp[kk * BS + cc];
                    }
                    sRhs[lane] -= sum;
                }
            #endif
            // sRhs -= sA * sYp

            __syncwarp();
        }

        // Y = DinvL[t] * rhs
        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            int rr = lane / BS;
            int cc = lane - rr * BS;
            VALUE_TYPE acc = 0.0;
            #pragma unroll
            for (int kk = 0; kk < BS; ++kk) {
                acc += inv[rr * BS + kk] * sRhs[kk * BS + cc];
            }
            Mval[(start + idx) * BS2 + lane] = acc;
        }
        __syncwarp();
    }
}
__device__ __forceinline__
void init_acc_identity_or_zero_5x5(int t, int j, int lane, double acc[2])
{
    int groupID           = lane >> 2;
    int threadID_in_group = lane &  3;

    int row = groupID;
    int c0  = threadID_in_group * 2 + 0;
    int c1  = threadID_in_group * 2 + 1;

    acc[0] = (t == j && row < 5 && c0 < 5 && row == c0) ? 1.0 : 0.0;
    acc[1] = (t == j && row < 5 && c1 < 5 && row == c1) ? 1.0 : 0.0;
}
__device__ __forceinline__
void tc_accumulate_rhs_5x5_k4(double acc[2], const double* sA, const double* Yp)
{
#if __CUDA_ARCH__ >= 800
    int lane = threadIdx.x & 31;

    double fragA = load_fragA_from_sA_5x5_first4cols(sA, lane);
    double fragB = load_fragB_from_Yp_5x5_first4rows(Yp, lane);

    // sRhs += (-A) * B
    fragA = -fragA;

    mma_m8n8k4_f64(acc, fragA, fragB);
#endif
}
__device__ __forceinline__
void add_tail_k1_into_acc_5x5(double acc[2], const double* sA, const double* Yp)
{
    int lane = threadIdx.x & 31;

    int groupID           = lane >> 2;
    int threadID_in_group = lane &  3;

    int row = groupID;
    int c0  = threadID_in_group * 2 + 0;
    int c1  = threadID_in_group * 2 + 1;

    if (row < 5) {
        double a_tail = sA[row * 5 + 4];   // A 的第 5 列

        if (c0 < 5) acc[0] -= a_tail * Yp[4 * 5 + c0];
        if (c1 < 5) acc[1] -= a_tail * Yp[4 * 5 + c1];
    }
}
__device__ __forceinline__
void scalar_update_rhs_5x5(double* sRhs, const double* sA, const double* sYp)
{
    int lane = threadIdx.x & 31;

    if (lane < 25) {
        int rr = lane / 5;
        int cc = lane % 5;

        double sum = 0.0;
        #pragma unroll
        for (int kk = 0; kk < 5; ++kk) {
            sum += sA[rr * 5 + kk] * sYp[kk * 5 + cc];
        }
        sRhs[rr * 5 + cc] -= sum;
    }
}
__device__ __forceinline__ VALUE_TYPE load_double_from_global_cs(const VALUE_TYPE* a)
{
    VALUE_TYPE r;
    asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(r) : "l"(a));
    return r;
}

__device__ __forceinline__ void store_double_to_global_cs(const VALUE_TYPE* a, VALUE_TYPE v)
{
    asm volatile("st.global.cs.f64 [%0], %1;" :: "l"(a), "d"(v));
}

__device__ __forceinline__ int load_int_from_global_cs(const int* a)
{
    int r;
    asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(r) : "l"(a));
    return r;
}
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_tensor_fix(
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
    const VALUE_TYPE* __restrict__ DinvL)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;

    int j = blockIdx.x * warpsPerBlock + warp;
    if (j >= nb) return;

    int start = load_int_from_global_cs(Mbrow + j);
    int end   = load_int_from_global_cs(Mbrow + j + 1);
    int N     = end - start;
    if (N <= 0) return;

    extern __shared__ VALUE_TYPE sh[];
    const int STRIDE = 3 * BS2; // rhs + Ablk + Yp
    VALUE_TYPE* base = sh + warp * STRIDE;
    VALUE_TYPE* sRhs = base + 0 * BS2;
    VALUE_TYPE* sA   = base + 1 * BS2;
    VALUE_TYPE* sYp  = base + 2 * BS2;

    for (int idx = 0; idx < N; ++idx) {
        int t = load_int_from_global_cs(Mcol + start + idx);

        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? 1.0 : 0.0;
        }
        __syncwarp();

        int g0   = start + idx;
        int qbeg = load_int_from_global_cs(PairPtr + g0);
        int qend = load_int_from_global_cs(PairPtr + g0 + 1);
        int qlen = qend - qbeg;


        for (int q = qbeg; q < qend; ++q) {
            int p_local = load_int_from_global_cs(PairPLocal + q);
            int srcA    = load_int_from_global_cs(PairSrc + q);

            if (lane < BS2) {
                sA[lane]  = load_double_from_global_cs(Abval + srcA * BS2 + lane);
                sYp[lane] = load_double_from_global_cs(Mval + (start + p_local) * BS2 + lane);
            }
            __syncwarp();
    
        #if __CUDA_ARCH__ >= 800
                tc_update_rhs_5x5_k4((double*)sRhs, (const double*)sA, (const double*)sYp);
                __syncwarp();
                add_tail_k4_outer_product((double*)sRhs, (const double*)sA, (const double*)sYp);

        #else
            scalar_update_rhs_5x5((double*)sRhs, (const double*)sA, (const double*)sYp);
        #endif

            __syncwarp();
        }

        if (lane < BS2) {
            int rr = lane / BS;
            int cc = lane - rr * BS;

            VALUE_TYPE acc = 0.0;
            #pragma unroll
            for (int kk = 0; kk < BS; ++kk) {
                VALUE_TYPE invv =
                    load_double_from_global_cs(DinvL + t * BS2 + rr * BS + kk);
                acc += invv * sRhs[kk * BS + cc];
            }

            store_double_to_global_cs(Mval + (start + idx) * BS2 + lane, acc);
        }
        __syncwarp();
    }
}

__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY_stream(
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
    int Nmax_perwarp              // 传入：每 warp 预留的最大列长
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

    // 防呆：如果某列 N > Nmax_perwarp，就直接返回或改成走 global（看你策略）
    if (N > Nmax_perwarp) return;

    // shared per-warp:
    //  sRhs(25) + sA(25) + Ycache(Nmax*25)
    extern __shared__ VALUE_TYPE sh[];
    VALUE_TYPE* warp_base = sh + warp * ( BS2 + 2*Nmax_perwarp*BS2 );

    VALUE_TYPE* sRhs   = warp_base + 0;
    VALUE_TYPE* sA     = warp_base + BS2;
    VALUE_TYPE* sYall  = warp_base + Nmax_perwarp*BS2;                 // [Nmax][25]

    // 可选：把缓存区清零（不是必须；我们会覆盖写每个 idx 的 25 元素）
    // if (lane < Nmax_perwarp*BS2) sYall[lane] = 0;
    // __syncwarp();

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        // rhs = (t==j ? I : 0)
        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }
        __syncwarp();

        // 用 Pair 列表累减：rhs -= sum A(t,p)*Yp
        int g0   = start + idx;
        int qbeg = PairPtr[g0];
        int qend = PairPtr[g0 + 1];
        int npairs = qend - qbeg;


            // 加载 Ablock 和 Yblock 到共享内存
            for (int q = 0; q < npairs; ++q) {
                int qq_load = qbeg + q;
                int p_local_load = PairPLocal[qq_load];
                int srcA_load = PairSrc[qq_load];

                // 读取 Ablock 到共享内存
                for (int e = lane; e < BS2; e += 32) {
                    sA[q * BS2 + e] = Abval[srcA_load * BS2 + e];
                }
            }

            __syncwarp();
        for (int q = qbeg; q < qend; ++q) {
            int p_local = PairPLocal[q];  // 0..idx-1
            int srcA    = PairSrc[q];

            int cur_idx=q-qbeg;
            if (lane < BS2) {
                // p_local 一定 < idx，因此一定已经写入 sYall
                // sYall 的布局：连续存 idx=0..N-1 的块
                // 每块 25 个
                // 取出第 p_local 块
                const VALUE_TYPE* Yp = sYall + p_local * BS2;
                // 为了保持你原逻辑，我们用 sRhs -= sA*Yp 直接算，不必再拷到 sYp 临时数组
                int rr = lane / BS;
                int cc = lane - rr * BS;
                VALUE_TYPE sum = 0;
                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    sum += sA[cur_idx*BS2+rr * BS + k] * Yp[k * BS + cc];
                }
                sRhs[lane] -= sum;
            }
            __syncwarp();
        }

        // Y = DinvL[t] * rhs（标量 5×5）
        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            int rr = lane / BS;
            int cc = lane - rr * BS;
            VALUE_TYPE acc = 0;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                acc += inv[rr * BS + k] * sRhs[k * BS + cc];
            }
            // 先写到 shared 缓存（本列内后续会用到）
            sYall[idx * BS2 + lane] = acc;
        }
        __syncwarp();
    }

    // 最后一次性把本列所有块写回 global（减少 global 往返）
    for (int idx = 0; idx < N; ++idx) {
        if (lane < BS2) {
            Mval[(start + idx) * BS2 + lane] = sYall[idx * BS2 + lane];
        }
        __syncwarp();
    }
}
__device__ __forceinline__
double load_A_from_concat_blocks(const VALUE_TYPE* sA,
                                 int npairs,
                                 int iter,
                                 int lane)
{
    if (lane >= 20) return 0.0;

    int row        = lane / 4;   // 0..4
    int col_in_itr = lane % 4;   // 0..3
    int global_col = iter * 4 + col_in_itr;

    int Ktotal = npairs * 5;
    if (global_col >= Ktotal) return 0.0;

    int block_id = global_col / 5;
    int col      = global_col % 5;

    return (double)sA[block_id * 25 + row * 5 + col];
}

// =============================ss===============================
// 从按 block 顺序存储的 Y blocks 中，按拼接后的 K 维取 B fragment
//
// B_concat 逻辑尺寸: (5*npairs) x 5
// TensorCore B tile: 4 x 8
//
// lane 映射假设:
//   row = lane / 8   (0..3)   -> k row inside this tile
//   col = lane % 8   (0..7)
//
// sPairPLocal[q] 给出第 q 个 pair 对应的 Y block 在 sYall 里的索引
// ============================================================
__device__ __forceinline__
double load_B_from_concat_blocks_col_major(const VALUE_TYPE* sYall,
                                           const int* sPairP,
                                           int npairs,
                                           int iter,
                                           int lane,
                                           int Nmax_perwarp)
{
    if (lane >= 20) return 0.0;

    int k_in_itr = lane % 4;
    int col      = lane / 4;

    if (col >= 5) return 0.0;

    int global_k = iter * 4 + k_in_itr;
    int Ktotal   = npairs * 5;
    if (global_k >= Ktotal) return 0.0;

    int block_id = global_k / 5;
    int row      = global_k % 5;

    if (block_id >= npairs) return 0.0;

    int p_local = sPairP[block_id];
    if (p_local < 0 || p_local >= Nmax_perwarp) return 0.0;

    const VALUE_TYPE* Yp = sYall + p_local * 25;
    return (double)Yp[row * 5 + col];
}



// ============================================================
// 把 fragC 累加结果减到 sRhs 上
//
// 这里采用的 C fragment 映射假设:
//   row  = lane / 4
//   col0 = (lane % 4) * 2
//   acc[0] -> (row, col0)
//   acc[1] -> (row, col0+1)
//
// 这个映射若和你的 simple example 不一致，需要只改这个函数。
// ============================================================
__device__ __forceinline__
void accum_fragC_sub_to_rhs(VALUE_TYPE* sRhs, int lane, const double* acc)
{
    int row  = lane >> 2;         // 0..7
    int col0 = (lane & 3) << 1;   // 0,2,4,6

    if (row < BS) {
        if (col0 < BS) {
            sRhs[row * BS + col0] -= (VALUE_TYPE)acc[0];
        }
        if (col0 + 1 < BS) {
            sRhs[row * BS + (col0 + 1)] -= (VALUE_TYPE)acc[1];
        }
    }
}

// ============================================================
// 你的 kernel：只把 pair-accumulate 那一段换成 Tensor Core
//
// 额外说明：
// 1. 这里修正了 sA 的 shared memory 大小，它必须至少能放 Nmax_perwarp 个 5x5 block
// 2. 新增一个 sPairP，用来缓存当前 idx 对应 pair 列表里的 p_local
// ============================================================
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY_stream_tensor(
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
    if (N > Nmax_perwarp) return;

    // --------------------------------------------------------
    // shared memory 布局（每 warp 一份）
    //
    // sRhs   : 25
    // sA     : Nmax_perwarp * 25      // 当前 idx 的所有 A_q
    // sYall  : Nmax_perwarp * 25      // 当前列已求出的 Y block
    // sPairP : Nmax_perwarp           // 当前 idx 的 pair 对应 p_local
    // --------------------------------------------------------
    extern __shared__ unsigned char shmem_raw[];

    size_t perWarpVals = BS2 + Nmax_perwarp * BS2 + Nmax_perwarp * BS2;
    size_t perWarpBytes = perWarpVals * sizeof(VALUE_TYPE)
                        + Nmax_perwarp * sizeof(int);

    unsigned char* warp_base = shmem_raw + warp * perWarpBytes;

    VALUE_TYPE* sRhs = reinterpret_cast<VALUE_TYPE*>(warp_base);
    warp_base += BS2 * sizeof(VALUE_TYPE);

    VALUE_TYPE* sA = reinterpret_cast<VALUE_TYPE*>(warp_base);
    warp_base += (size_t)Nmax_perwarp * BS2 * sizeof(VALUE_TYPE);

    VALUE_TYPE* sYall = reinterpret_cast<VALUE_TYPE*>(warp_base);
    warp_base += (size_t)Nmax_perwarp * BS2 * sizeof(VALUE_TYPE);

    int* sPairP = reinterpret_cast<int*>(warp_base);

    // --------------------------------------------------------
    // 主循环：按列内 idx 顺序递推
    // --------------------------------------------------------
    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        // rhs = (t==j ? I : 0)
        if (lane < BS2) {
            int r = lane / BS;
            int c = lane % BS;
            sRhs[lane] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }
        __syncwarp();

        int g0     = start + idx;
        int qbeg   = PairPtr[g0];
        int qend   = PairPtr[g0 + 1];
        int npairs = qend - qbeg;

        // ----------------------------------------------------
        // 先把当前 idx 需要的所有 A block 和对应 p_local 缓存到 shared
        // sA[q]     = A_q
        // sPairP[q] = p_local(q)
        // ----------------------------------------------------
        for (int q = 0; q < npairs; ++q) {
            int qq = qbeg + q;
            int p_local = PairPLocal[qq];
            int srcA    = PairSrc[qq];

            if (lane == 0) {
                sPairP[q] = p_local;
            }

            for (int e = lane; e < BS2; e += 32) {
                sA[q * BS2 + e] = Abval[srcA * BS2 + e];
            }
        }
        __syncwarp();

        // ----------------------------------------------------
        // 用 Tensor Core 计算:
        // rhs -= sum_q A_q * Y_{p_local(q)}
        //
        // 即做一次:
        //   [A0|A1|...|An] * [Y0;Y1;...;Yn]
        // K 总长度 = npairs * 5
        // 所以 m8n8k4 次数 = ceil(npairs*5 / 4)
        // ----------------------------------------------------
        if (npairs > 4) {
            // 使用 Tensor Core 计算
            double acc[2] = {0.0, 0.0};
            int ktiles = (npairs * BS + 3) / 4;

            #pragma unroll
            for (int ktile = 0; ktile < ktiles; ++ktile) {
                double fragA = load_A_from_concat_blocks(sA, npairs, ktile, lane);
                double fragB = load_B_from_concat_blocks_col_major(sYall, sPairP, npairs, ktile, lane, Nmax_perwarp);
                mma_m8n8k4_f64(acc, fragA, fragB);
            }

            accum_fragC_sub_to_rhs(sRhs, lane, acc);
        } else if (npairs > 0) {
            // 使用原标量方法计算
            for (int q = qbeg; q < qend; ++q) {
                int p_local = PairPLocal[q];
                int cur_idx = q - qbeg;

                if (lane < BS2) {
                    const VALUE_TYPE* Yp = sYall + p_local * BS2;
                    int rr = lane / BS;
                    int cc = lane % BS;
                    VALUE_TYPE sum = 0;
                    #pragma unroll
                    for (int k = 0; k < BS; ++k) {
                        sum += sA[cur_idx * BS2 + rr * BS + k] * Yp[k * BS + cc];
                    }
                    sRhs[lane] -= sum;
                }
                __syncwarp();
            }
        }


        // ----------------------------------------------------
        // 后面的 DinvL * rhs 仍然保留你原来的标量版本
        // ----------------------------------------------------
        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            int rr = lane / BS;
            int cc = lane % BS;
            VALUE_TYPE acc = 0;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                acc += inv[rr * BS + k] * sRhs[k * BS + cc];
            }
            sYall[idx * BS2 + lane] = acc;
        }
        __syncwarp();
    }

    // --------------------------------------------------------
    // 最后一次性写回 global
    // --------------------------------------------------------
    for (int idx = 0; idx < N; ++idx) {
        if (lane < BS2) {
            Mval[(start + idx) * BS2 + lane] = sYall[idx * BS2 + lane];
        }
        __syncwarp();
    }
}

// 约定：本内核使用 float 数据（A、Y、DinvL、Mval 全是 float）。
// 若你工程里还有 double 版本，请单独保留之前的纯标量核用于 double。

__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_cachedY_tc_two_per_batch(
    int nb,
    const int*  __restrict__ Abrow,     // 未用（保留）
    const int*  __restrict__ Abcol,     // 未用（保留）
    const float*__restrict__ Abval,     // A 的块值 float，每块25
    const int*  __restrict__ Mbrow,     // per-column rowptr
    const int*  __restrict__ Mcol,      // S_j
    const int*  __restrict__ PairPtr,   // 对列表 rowptr
    const int*  __restrict__ PairPLocal,// p 的列内局部号（<idx）
    const int*  __restrict__ PairSrc,   // A 中 (t,p) 的块号
    float*      __restrict__ Mval,      // 输出（BCSC），每块25
    const float*__restrict__ DinvL,     // 预逆 nb*25
    int Nmax_perwarp                    // 列内缓存大小上限
){
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warpsPerBlock = blockDim.x >> 5;

    const int j = blockIdx.x * warpsPerBlock + warp; // warp -> column
    if (j >= nb) return;

    const int start = Mbrow[j];
    const int end   = Mbrow[j + 1];
    const int N     = end - start;
    if (N <= 0) return;
    if (N > Nmax_perwarp) return; // 超过缓存上限：你也可以改成回退标量

    // ---------------- shared 对齐切片 ----------------
    // 动态 shared 用字节声明，保证起点 16B 对齐
    extern __shared__ __align__(128) unsigned char sbytes[]; // 直接 128B 对齐更保险
    // 我们用 “按 float 元素计数”的布局 + 128B 边界对齐
    const int ALIGN_WORDS = 128 / sizeof(float);   // 128B 对齐对应的 float 元素数

    // 每 warp 的布局： [sRhs(25)] [sA(25)] [sYcache(N*25)] [pad到128B] [A16(256h)] [B16(256h)] [C16(256f)]
    int words_small   = 2 * BS2;             // 50
    int words_cache   = Nmax_perwarp * BS2;  // 25*Nmax
    int words_before  = words_small + words_cache;
    int pad_words     = (ALIGN_WORDS - (words_before % ALIGN_WORDS)) & (ALIGN_WORDS - 1); // 让瓦片起点落在128B
    int words_tiles   = 3 * 16 * 16;         // 3*256 = 768 float
    int words_perwarp = words_before + pad_words + words_tiles;

    float* fbase = reinterpret_cast<float*>(sbytes);
    float* warp_base = fbase + warp * words_perwarp;

    float* sRhs  = warp_base + 0;         // [25]
    float* sA    = warp_base + BS2;       // [25]（TC 路径用不到也无妨）
    float* sYall = warp_base + 2*BS2;     // [Nmax*25]

    // 瓦片起点（已做128B对齐）
    float* tiles = warp_base + words_before + pad_words;
    half*  sA16h = reinterpret_cast<half*>( tiles + 0 * 16 * 16 ); // 256 half
    half*  sB16h = reinterpret_cast<half*>( tiles + 1 * 16 * 16 ); // 256 half
    float* sC16f =                ( tiles + 2 * 16 * 16 );         // 256 float

    const int rr = lane / BS;
    const int cc = lane % BS;

    const unsigned FULL = 0xFFFFFFFFu;
    const unsigned act  = __ballot_sync(FULL, lane < BS2);

    // 将 5x5 float 行写入半精 tile 的 (r0,c0)；只有在 16B 对齐时才用 float4 读
    auto write5x5_half = [&](const float* __restrict__ src, half* __restrict__ tile, int r0, int c0){
        if (lane < BS) {
            int r = lane;
            const float* row = src + r * BS;
            float a0,a1,a2,a3,a4;
            uintptr_t addr = reinterpret_cast<uintptr_t>(row);
            if ((addr & 0xF) == 0) {
                // 16B 对齐才用 float4（shared/global 都 OK）
                float4 v4 = *reinterpret_cast<const float4*>(row);
                a0=v4.x; a1=v4.y; a2=v4.z; a3=v4.w; a4=row[4];
            } else {
                a0=row[0]; a1=row[1]; a2=row[2]; a3=row[3]; a4=row[4];
            }
            int base = (r0 + r) * 16 + c0;
            tile[base + 0] = __float2half_rn(a0);
            tile[base + 1] = __float2half_rn(a1);
            tile[base + 2] = __float2half_rn(a2);
            tile[base + 3] = __float2half_rn(a3);
            tile[base + 4] = __float2half_rn(a4);
        }
        __syncwarp(act);
    };

    // 清 5x5 半精 tile 的 (r0,c0)
    auto zero5x5_half = [&](half* __restrict__ tile, int r0, int c0){
        if (lane < BS) {
            int r = lane;
            int base = (r0 + r) * 16 + c0;
            tile[base + 0] = __float2half_rn(0.f);
            tile[base + 1] = __float2half_rn(0.f);
            tile[base + 2] = __float2half_rn(0.f);
            tile[base + 3] = __float2half_rn(0.f);
            tile[base + 4] = __float2half_rn(0.f);
        }
        __syncwarp(act);
    };

    for (int idx = 0; idx < N; ++idx) {
        const int t = Mcol[start + idx];

        // rhs = (t==j ? I : 0)
        if (lane < BS2) {
            sRhs[lane] = (t == j && rr == cc) ? 1.f : 0.f;
        }
        __syncwarp(act);

        // 初始化瓦片（一次/idx），Cf 置 0
        for (int e = lane; e < 16*16; e += 32) {
            sA16h[e] = __float2half_rn(0.f);
            sB16h[e] = __float2half_rn(0.f);
        }
        __syncwarp(act);

        wmma::fragment<wmma::accumulator, 16,16,16, float> Cf;
        wmma::fragment<wmma::matrix_a,    16,16,16, half,  wmma::row_major> Af;
        wmma::fragment<wmma::matrix_b,    16,16,16, half,  wmma::row_major> Bf;
        wmma::fill_fragment(Cf, 0.0f);

        const int g0   = start + idx;
        const int qbeg = PairPtr[g0];
        const int qend = PairPtr[g0 + 1];

        int qb = qbeg;
        for (; qb + 1 < qend; qb += 2) {
            // pair0 -> TL(0,0)
            {
                const int p0 = PairPLocal[qb + 0];
                const int s0 = PairSrc   [qb + 0];
                const float* A0 = Abval + s0 * BS2;
                const float* Y0 = sYall + p0 * BS2;   // cachedY
                write5x5_half(A0, sA16h, 0, 0);
                write5x5_half(Y0, sB16h, 0, 0);
            }
            // pair1 -> BR(8,8)
            {
                const int p1 = PairPLocal[qb + 1];
                const int s1 = PairSrc   [qb + 1];
                const float* A1 = Abval + s1 * BS2;
                const float* Y1 = sYall + p1 * BS2;
                write5x5_half(A1, sA16h, 8, 8);
                write5x5_half(Y1, sB16h, 8, 8);
            }

            wmma::load_matrix_sync(Af, sA16h, 16);
            wmma::load_matrix_sync(Bf, sB16h, 16);
            wmma::mma_sync(Cf, Af, Bf, Cf);
            __syncwarp(act);
        }

        // 零头1对：只填 TL，BR 显式清 5x5
        if (qb < qend) {
            const int p = PairPLocal[qb];
            const int s = PairSrc   [qb];
            const float* A = Abval + s * BS2;
            const float* Y = sYall + p * BS2;
            write5x5_half(A, sA16h, 0, 0);
            write5x5_half(Y, sB16h, 0, 0);
            zero5x5_half(sA16h, 8, 8);
            zero5x5_half(sB16h, 8, 8);

            wmma::load_matrix_sync(Af, sA16h, 16);
            wmma::load_matrix_sync(Bf, sB16h, 16);
            wmma::mma_sync(Cf, Af, Bf, Cf);
            __syncwarp(act);
        }

        // 末尾一次 store，再抽取 TL/BR 的 5×5
        wmma::store_matrix_sync(sC16f, Cf, 16, wmma::mem_row_major);
        if (lane < BS2) {
            float acc = sC16f[rr*16 + cc] + sC16f[(8 + rr)*16 + (8 + cc)];
            sRhs[lane] -= acc;
        }
        __syncwarp(act);

        // 标量 5×5：Y = DinvL[t] * sRhs
        if (lane < BS2) {
            const float* inv = DinvL + t * BS2;
            float y = 0.f;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                y += inv[rr * BS + k] * sRhs[k * BS + cc];
            }
            sYall[idx * BS2 + lane] = y; // 写缓存
        }
        __syncwarp(act);
    }

    // 整列写回
    for (int idx = 0; idx < N; ++idx) {
        if (lane < BS2) {
            Mval[(start + idx) * BS2 + lane] = sYall[idx * BS2 + lane];
        }
        __syncwarp(act);
    }
}


// 计数：为每个 (列 j, 列内 idx) 统计命中的 (t, p_local) 个数（p_local < idx 且 (t,p)∈A）
__global__ void count_pairs_lower_kernel_v2(
    int nb,
    const int* __restrict__ Abrow,   // len nb+1
    const int* __restrict__ Abcol,   // len nnzb
    const int* __restrict__ Mbrow,   // len nb+1
    const int* __restrict__ Mcol,    // len Mnnz
    int* __restrict__ PairPtr_counts // len Mnnz+1, 此处暂存"计数"；最后一个元素我们在 host 置零
){
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;
    int j = blockIdx.x * warpsPerBlock + warp;
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j+1];
    int N     = end - start;
    if (N <= 0) return;

    // 每个列内条目 idx 都有一段 pairs
    for (int idx = 0; idx < N; ++idx) {
        if (lane == 0) {
            int t = Mcol[start + idx];   // 该行
            int a0 = Abrow[t];
            int a1 = Abrow[t+1];
            int iA = a0;
            int iS = start;              // 注意：绝对下标遍历 [start, end)
            int cnt = 0;

            while (iA < a1 && iS < end) {
                int colA = Abcol[iA];      // A 的列号（全局）
                int colS = Mcol[iS];       // S_j 的一个元素（全局行号，同时也是"列号"）
                if (colA == colS) {
                    // 只接受 p_local < idx
                    if (iS - start < idx) ++cnt;
                    ++iA; ++iS;
                } else if (colA < colS) {
                    ++iA;
                } else {
                    ++iS;
                }
            }
            PairPtr_counts[start + idx] = cnt;
        }
        __syncwarp();
    }
}

// 填充：写 PairPtr(前缀后)、PairPLocal、PairSrc
__global__ void fill_pairs_lower_kernel_v2(
    int nb,
    const int* __restrict__ Abrow,
    const int* __restrict__ Abcol,
    const int* __restrict__ Mbrow,
    const int* __restrict__ Mcol,
    const int* __restrict__ PairPtr,     // len Mnnz+1, 已 exclusive_scan
    int* __restrict__ PairPLocal,        // len total_pairs
    int* __restrict__ PairSrc            // len total_pairs
){
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;
    int j = blockIdx.x * warpsPerBlock + warp;
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j+1];
    int N     = end - start;
    if (N <= 0) return;

    for (int idx = 0; idx < N; ++idx) {
        if (lane == 0) {
            int t = Mcol[start + idx];
            int a0 = Abrow[t], a1 = Abrow[t+1];
            int iA = a0;
            int iS = start;

            int out = PairPtr[start + idx]; // 本 (j,idx) 的 pairs 起点
            while (iA < a1 && iS < end) {
                int colA = Abcol[iA];
                int colS = Mcol[iS];
                if (colA == colS) {
                    int p_local = iS - start;    // 0..N-1
                    if (p_local < idx) {
                        PairPLocal[out] = p_local; // 列内局部号
                        PairSrc[out]    = iA;      // A 的 BSR 位置
                        ++out;
                    }
                    ++iA; ++iS;
                } else if (colA < colS) {
                    ++iA;
                } else {
                    ++iS;
                }
            }
        }
        __syncwarp();
    }
}


// BCSC(block, BS=5) * 向量： y += M * x  =====
__global__ void bcsc_matvec_add_bsr5(
    int nb,
    const int* __restrict__ Mbrow,   // 列指针（nb+1）
    const int* __restrict__ Mcol,    // 行索引（nnzb）
    const VALUE_TYPE* __restrict__ Mval, // 块值（nnzb*BS2），按 BCSC 顺序
    const VALUE_TYPE* __restrict__ x,    // 输入（长度 nb*BS）
    VALUE_TYPE* __restrict__ y)          // 输出（长度 nb*BS），外部先清零
{
    int j = blockIdx.x;              // 列块
    int tid = threadIdx.x;           // 该列内的第 tid 个非零块
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N == 0 || tid >= N) return;

    int i = Mcol[start + tid];                      // 行块
    const VALUE_TYPE* Bij = Mval + (start + tid) * BS2; // 5x5 块
    const VALUE_TYPE* xj  = x + j * BS;                 // 列块向量

    // 该块对 y_i 的 5 维贡献
    VALUE_TYPE acc[BS];
    #pragma unroll
    for (int r = 0; r < BS; ++r) {
        VALUE_TYPE s = 0.0;
        #pragma unroll
        for (int c = 0; c < BS; ++c) {
            s += Bij[r * BS + c] * xj[c];
        }
        acc[r] = s;
    }
    // 原子加到 y 的目标块
    #pragma unroll
    for (int r = 0; r < BS; ++r) {
        atomicAdd(&y[i * BS + r], acc[r]);
    }
}
// 仅调试：一列一发，不用原子（因为没有跨列并发）
// 这里 blockDim 至少 >= 本列 N（但你 maxN=12，随便 32/64 足够）
__global__ void bcsc_matvec_onecol_noatomic(
    int j,
    int nb,
    const int* __restrict__ Mbrow,
    const int* __restrict__ Mcol,
    const VALUE_TYPE* __restrict__ Mval,
    const VALUE_TYPE* __restrict__ x,
    VALUE_TYPE* __restrict__ y)
{
    const int BS=5, BS2=25;
    int tid = threadIdx.x;
    if (j >= nb) return;
    int start = Mbrow[j], end = Mbrow[j+1];
    int N = end - start;
    if (tid >= N) return;

    int i = Mcol[start + tid];
    const VALUE_TYPE* Bij = Mval + (start + tid) * BS2;
    const VALUE_TYPE* xj  = x + j * BS;

    VALUE_TYPE acc[BS] = {0,0,0,0,0};
    #pragma unroll
    for (int r = 0; r < BS; ++r) {
        VALUE_TYPE s = 0.0;
        #pragma unroll
        for (int c = 0; c < BS; ++c)
            s += Bij[r*BS + c] * xj[c];
        acc[r] = s;
    }
    VALUE_TYPE* yi = y + i * BS;
    #pragma unroll
    for (int r = 0; r < BS; ++r) yi[r] += acc[r];  // 无原子
}

inline void launch_bcsc_matvec_add_bsr5_serial_debug(
    int nb,
    const int* d_Mbrow, const int* d_Mcol, const VALUE_TYPE* d_Mval,
    const VALUE_TYPE* d_x, VALUE_TYPE* d_y)
{
    // y 外面请先清零
    for (int j=0; j<nb; ++j) {
        int h_start, h_end;
        cudaMemcpy(&h_start, d_Mbrow + j,     sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_end,   d_Mbrow + j + 1, sizeof(int), cudaMemcpyDeviceToHost);
        int N = h_end - h_start;
        int blk = (N>0 ? ((N+31)/32)*32 : 32);
        bcsc_matvec_onecol_noatomic<<<1, blk>>>(j, nb, d_Mbrow, d_Mcol, d_Mval, d_x, d_y);
        cudaDeviceSynchronize();
    }
}

// 简单 launcher（可选）
inline void launch_bcsc_matvec_add_bsr5(
    int nb,
    const int* d_Mbrow, const int* d_Mcol, const VALUE_TYPE* d_Mval,
    const VALUE_TYPE* d_x, VALUE_TYPE* d_y, cudaStream_t stream = 0)
{
    dim3 grid(nb), block(128);
    bcsc_matvec_add_bsr5<<<grid, block, 0, stream>>>(
        nb, d_Mbrow, d_Mcol, d_Mval, d_x, d_y);
}


// 打印一个 BCSC(block) 矩阵
void print_bcsc_blocks(const char* name,
                       int nb,
                       const std::vector<int>& Mbrow,
                       const std::vector<int>& Mcol,
                       const std::vector<VALUE_TYPE>& Mval)
{
    std::cout << name << " (block " << nb << "x" << nb << ", block size 5):\n";
    for (int j = 0; j < nb; ++j) {
        int start = Mbrow[j];
        int end   = Mbrow[j + 1];
        for (int idx = start; idx < end; ++idx) {
            int i = Mcol[idx];
            const VALUE_TYPE* blk = &Mval[idx * BS2];
            std::cout << "  block (" << i << "," << j << "):\n";
            for (int r = 0; r < BS; ++r) {
                std::cout << "    ";
                for (int c = 0; c < BS; ++c) {
                    std::cout << blk[r * BS + c] << " ";
                }
                std::cout << "\n";
            }
        }
    }
}
/*
// ============================================================
// f64 mma.sync m16n8k16
// D = A * B + C
// A: 8 regs / thread
// B: 4 regs / thread
// C,D: 4 regs / thread
// ============================================================
__device__ __forceinline__
void mma_m16n8k16_f64(
    double D[4],
    const double A[8],
    const double B[4],
    const double C[4])
{
#if __CUDA_ARCH__ >= 800
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f64.f64.f64.f64 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7, %8, %9, %10, %11}, "
        "{%12, %13, %14, %15}, "
        "{%16, %17, %18, %19};\n"
        : "=d"(D[0]), "=d"(D[1]), "=d"(D[2]), "=d"(D[3])
        : "d"(A[0]), "d"(A[1]), "d"(A[2]), "d"(A[3]),
          "d"(A[4]), "d"(A[5]), "d"(A[6]), "d"(A[7]),
          "d"(B[0]), "d"(B[1]), "d"(B[2]), "d"(B[3]),
          "d"(C[0]), "d"(C[1]), "d"(C[2]), "d"(C[3]));
#else
    D[0] = C[0];
    D[1] = C[1];
    D[2] = C[2];
    D[3] = C[3];
#endif
}


// ============================================================
// A_panel = [A0 A1 A2], size = 5x15, logical pad to 16x16
// row-major
//
// PTX f64 A-fragment mapping for m16n8k16:
// groupID = lane >> 2
// tid4    = lane & 3
//
// row = groupID           if i % 2 == 0
//       groupID + 8       otherwise
//
// col = i*2 + tid4        if i % 2 == 0
//       i*2 - 2 + tid4    otherwise
// ============================================================
__device__ __forceinline__
double panelA_5x15_get(
    const VALUE_TYPE* __restrict__ A0,
    const VALUE_TYPE* __restrict__ A1,
    const VALUE_TYPE* __restrict__ A2,
    int row, int col)
{
    if (row >= 5 || col >= 15) return 0.0;

    const VALUE_TYPE* Ab = (col < 5) ? A0 : ((col < 10) ? A1 : A2);
    int lc = col % 5;
    return (double)Ab[row * 5 + lc];
}

__device__ __forceinline__
void fill_fragA_from_panel_5x15_f64(
    int lane,
    const VALUE_TYPE* __restrict__ A0,
    const VALUE_TYPE* __restrict__ A1,
    const VALUE_TYPE* __restrict__ A2,
    double fragA[8])
{
    int groupID = lane >> 2;
    int tid4    = lane & 3;

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        int row = (i & 1) ? (groupID + 8) : groupID;
        int col = (i & 1) ? ((i << 1) - 2 + tid4)
                          : ((i << 1) + tid4);
        fragA[i] = panelA_5x15_get(A0, A1, A2, row, col);
    }
}


// ============================================================
// B_panel = [Y0; Y1; Y2], size = 15x5, logical pad to 16x8
// B is col-major semantic in mma.row.col
//
// PTX f64 B-fragment mapping for m16n8k16:
// groupID = lane >> 2
// tid4    = lane & 3
//
// row = tid4 + 4*i
// col = groupID
// ============================================================
__device__ __forceinline__
double panelB_15x5_get(
    const VALUE_TYPE* __restrict__ Y0,
    const VALUE_TYPE* __restrict__ Y1,
    const VALUE_TYPE* __restrict__ Y2,
    int row, int col)
{
    if (row >= 15 || col >= 5) return 0.0;

    const VALUE_TYPE* Yb = (row < 5) ? Y0 : ((row < 10) ? Y1 : Y2);
    int lr = row % 5;
    return (double)Yb[lr * 5 + col];
}

__device__ __forceinline__
void fill_fragB_from_panel_15x5_f64(
    int lane,
    const VALUE_TYPE* __restrict__ Y0,
    const VALUE_TYPE* __restrict__ Y1,
    const VALUE_TYPE* __restrict__ Y2,
    double fragB[4])
{
    int groupID = lane >> 2;
    int tid4    = lane & 3;

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = tid4 + 4 * i;
        int col = groupID;
        fragB[i] = panelB_15x5_get(Y0, Y1, Y2, row, col);
    }
}


// ============================================================
// Accumulator mapping for f64 m16n8k16
//
// row = groupID         if i < 2
//       groupID + 8     if i >= 2
// col = 2*tid4 + (i&1)
// ============================================================
__device__ __forceinline__
void acc_rc_to_lane_reg_f64_m16n8k16(
    int r, int c,
    int &owner_lane,
    int &owner_reg)
{
    int groupID = (r < 8) ? r : (r - 8);
    int tid4    = c >> 1;
    int odd     = c & 1;

    owner_lane = (groupID << 2) + tid4;
    owner_reg  = ((r < 8) ? 0 : 2) + odd;
}

__device__ __forceinline__
double fetch_c_rc_from_fragD_f64_m16n8k16(
    int lane,
    const double fragD[4],
    int r, int c)
{
    unsigned mask = 0xffffffffu;

    int owner_lane, owner_reg;
    acc_rc_to_lane_reg_f64_m16n8k16(r, c, owner_lane, owner_reg);

    double v0 = __shfl_sync(mask, fragD[0], owner_lane);
    double v1 = __shfl_sync(mask, fragD[1], owner_lane);
    double v2 = __shfl_sync(mask, fragD[2], owner_lane);
    double v3 = __shfl_sync(mask, fragD[3], owner_lane);

    if (owner_reg == 0) return v0;
    if (owner_reg == 1) return v1;
    if (owner_reg == 2) return v2;
    return v3;
}


// ============================================================
// 标量 fallback：rhs(rr,cc) -= A*Y 对应元素
// ============================================================
__device__ __forceinline__
void scalar_sub_one_entry_5x5(
    int lane,
    VALUE_TYPE &rhs_reg,
    const VALUE_TYPE* __restrict__ A25,
    const VALUE_TYPE* __restrict__ Y25)
{
    if (lane < BS2) {
        int rr = lane / BS;
        int cc = lane % BS;

        VALUE_TYPE sum = 0.0;
#pragma unroll
        for (int k = 0; k < BS; ++k) {
            sum += A25[rr * BS + k] * Y25[k * BS + cc];
        }
        rhs_reg -= sum;
    }
}


// ============================================================
// chunk=3 Tensor Core:
// rhs(rr,cc) -= ([A0 A1 A2] * [Y0;Y1;Y2])(rr,cc)
// ============================================================
__device__ __forceinline__
void tc_sub_chunk3_one_entry_f64_direct_frag(
    int lane,
    VALUE_TYPE &rhs_reg,
    const VALUE_TYPE* __restrict__ A0,
    const VALUE_TYPE* __restrict__ A1,
    const VALUE_TYPE* __restrict__ A2,
    const VALUE_TYPE* __restrict__ Y0,
    const VALUE_TYPE* __restrict__ Y1,
    const VALUE_TYPE* __restrict__ Y2)
{
#if __CUDA_ARCH__ >= 800
    double fragA[8];
    double fragB[4];
    double fragC[4] = {0.0, 0.0, 0.0, 0.0};
    double fragD[4];

    fill_fragA_from_panel_5x15_f64(lane, A0, A1, A2, fragA);
    fill_fragB_from_panel_15x5_f64(lane, Y0, Y1, Y2, fragB);
    mma_m16n8k16_f64(fragD, fragA, fragB, fragC);

    if (lane < BS2) {
        int rr = lane / BS;
        int cc = lane % BS;

        double tc_val = fetch_c_rc_from_fragD_f64_m16n8k16(lane, fragD, rr, cc);
        rhs_reg -= (VALUE_TYPE)tc_val;
    }
#else
    if (lane < BS2) {
        int rr = lane / BS;
        int cc = lane % BS;

        VALUE_TYPE sum = 0.0;
#pragma unroll
        for (int k = 0; k < BS; ++k) {
            sum += A0[rr * BS + k] * Y0[k * BS + cc];
            sum += A1[rr * BS + k] * Y1[k * BS + cc];
            sum += A2[rr * BS + k] * Y2[k * BS + cc];
        }
        rhs_reg -= sum;
    }
#endif
}


// ============================================================
// Y(rr,cc) = Dinv(t) * rhs
// rhs 仍然是 lane<25 的 row-major 单元素寄存器布局
// ============================================================
__device__ __forceinline__
VALUE_TYPE compute_y_one_entry_from_rhs_reg(
    int lane,
    VALUE_TYPE rhs_reg,
    const VALUE_TYPE* __restrict__ Dinv_block)
{
    if (lane >= BS2) return (VALUE_TYPE)0.0;

    unsigned mask = 0xffffffffu;

    int rr = lane / BS;
    int cc = lane % BS;

    VALUE_TYPE acc = 0.0;
#pragma unroll
    for (int k = 0; k < BS; ++k) {
        int src_lane = k * BS + cc;
        VALUE_TYPE rhs_kc = __shfl_sync(mask, rhs_reg, src_lane);
        acc += Dinv_block[rr * BS + k] * rhs_kc;
    }
    return acc;
}


// ============================================================
// 完整 kernel（简化版）
// - 不用 shared panel
// - 不用 shared Y cache
// - Y(idx) 算完立即写 Mval
// - 后续 Pair 直接从 Mval 读 Yp
// ============================================================
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_tc_directfrag_f64(
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
    const VALUE_TYPE* __restrict__ DinvL)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;

    int j = blockIdx.x * warpsPerBlock + warp;
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N <= 0) return;

    VALUE_TYPE rhs_reg = 0.0;

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        // rhs = I or 0
        if (lane < BS2) {
            int rr = lane / BS;
            int cc = lane % BS;
            rhs_reg = (t == j && rr == cc) ? (VALUE_TYPE)1.0 : (VALUE_TYPE)0.0;
        }

        int g0   = start + idx;
        int qbeg = PairPtr[g0];
        int qend = PairPtr[g0 + 1];

        int q = qbeg;

#if __CUDA_ARCH__ >= 800
        // 3个一组：Tensor Core
        for (; q + 2 < qend; q += 3) {
            int p0 = PairPLocal[q + 0];
            int p1 = PairPLocal[q + 1];
            int p2 = PairPLocal[q + 2];

            int a0 = PairSrc[q + 0];
            int a1 = PairSrc[q + 1];
            int a2 = PairSrc[q + 2];

            const VALUE_TYPE* A0 = Abval + a0 * BS2;
            const VALUE_TYPE* A1 = Abval + a1 * BS2;
            const VALUE_TYPE* A2 = Abval + a2 * BS2;

            const VALUE_TYPE* Y0 = Mval + (start + p0) * BS2;
            const VALUE_TYPE* Y1 = Mval + (start + p1) * BS2;
            const VALUE_TYPE* Y2 = Mval + (start + p2) * BS2;

            tc_sub_chunk3_one_entry_f64_direct_frag(
                lane, rhs_reg,
                A0, A1, A2,
                Y0, Y1, Y2
            );
        }
#endif

        // 剩余不足3个：标量
        for (; q < qend; ++q) {
            int p_local = PairPLocal[q];
            int srcA    = PairSrc[q];

            const VALUE_TYPE* Ablk = Abval + srcA * BS2;
            const VALUE_TYPE* Yblk = Mval + (start + p_local) * BS2;

            scalar_sub_one_entry_5x5(lane, rhs_reg, Ablk, Yblk);
        }

        // Y = DinvL[t] * rhs
        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            VALUE_TYPE y_reg = compute_y_one_entry_from_rhs_reg(lane, rhs_reg, inv);

            // 立即写回，后续 pair 可直接读取
            Mval[(start + idx) * BS2 + lane] = y_reg;
        }
    }
}*/

template <typename T>
__device__ __forceinline__
void emulate_m8n8k4_fp64(
    const T a_frag[8][4],   // A: [8 x 4]
    const T b_frag[4][8],   // B: [4 x 8]
    T c_frag[8][8]          // C: [8 x 8], accumulate
){
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            T sum = c_frag[i][j];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                sum += a_frag[i][k] * b_frag[k][j];
            }
            c_frag[i][j] = sum;
        }
    }
}

// ------------------------------------------------------------
// 计算：Csum = sum_q A_q * Y_q
//
// 其中：
//   A_q, Y_q 都是 5x5
//   q = 0..npairs-1
//
// 但是内部不按 block 做，而是按你说的方式：
//   A_panel: 5 x (5*npairs)
//   B_panel: (5*npairs) x 5
//
// 然后沿着 k 维每次取 4 列，做一次“m8n8k4风格”的微核
//
// 注意：为了让逻辑最清楚，这里让 lane 0 执行整个微核，结果写到 shared 的 Csum。
//       这是“结构正确版”，不是最终高性能版。
// ------------------------------------------------------------
template <typename T>
__device__ __forceinline__
void microkernel_sum_pairs_streamed(
    int lane,
    int npairs,
    const T* __restrict__ Ablocks_pack, // [npairs][25]
    const T* __restrict__ Yblocks_pack, // [npairs][25]
    T* __restrict__ A_panel,            // [5][5*npairs]
    T* __restrict__ B_panel,            // [5*npairs][5]
    T* __restrict__ Csum                // [25]
){
    const int Ktot = BS * npairs; // 总“流式列数”

    // 1) 先把所有 A_q / Y_q 打包成 panel
    // A_panel[r][5*q + c] = A_q(r,c)
    // B_panel[5*q + r][c] = Y_q(r,c)

    int totalA = BS * Ktot; // 5 * (5*npairs)
    for (int idx = lane; idx < totalA; idx += 32) {
        int r = idx / Ktot;   // 0..4
        int k = idx % Ktot;   // 0..5*npairs-1
        int q = k / BS;       // 第几个 block
        int c = k % BS;       // block 内第几列
        A_panel[idx] = Ablocks_pack[q * BS2 + r * BS + c];
    }

    int totalB = Ktot * BS;   // (5*npairs) * 5
    for (int idx = lane; idx < totalB; idx += 32) {
        int k = idx / BS;     // 0..5*npairs-1
        int c = idx % BS;     // 0..4
        int q = k / BS;       // 第几个 block
        int r = k % BS;       // block 内第几行
        B_panel[idx] = Yblocks_pack[q * BS2 + r * BS + c];
    }
    __syncwarp();

    // 2) 只有 lane 0 做完整的 8x8 microkernel（先保证结构正确）
    if (lane == 0) {
        T c_frag[8][8];

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                c_frag[i][j] = (T)0;
            }
        }

        // 按 k=4 切片
        for (int k0 = 0; k0 < Ktot; k0 += 4) {
            T a_frag[8][4];
            T b_frag[4][8];

            // 装 A fragment: [8 x 4]
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                #pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    int gk = k0 + kk;
                    if (i < BS && gk < Ktot) {
                        a_frag[i][kk] = A_panel[i * Ktot + gk];
                    } else {
                        a_frag[i][kk] = (T)0;
                    }
                }
            }

            // 装 B fragment: [4 x 8]
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                int gk = k0 + kk;
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    if (j < BS && gk < Ktot) {
                        B_frag:
                        b_frag[kk][j] = B_panel[gk * BS + j];
                    } else {
                        b_frag[kk][j] = (T)0;
                    }
                }
            }

            emulate_m8n8k4_fp64(a_frag, b_frag, c_frag);
        }

        // 只把左上 5x5 写回
        #pragma unroll
        for (int r = 0; r < BS; ++r) {
            #pragma unroll
            for (int c = 0; c < BS; ++c) {
                Csum[r * BS + c] = c_frag[r][c];
            }
        }
    }

    __syncwarp();
}

// ------------------------------------------------------------
// 完整 kernel：在你的原 kernel 上改
//
// 新增参数：
//   maxPairs_perwarp
//
// 作用：
//   每个 warp 预留最多多少个 pair 的缓存空间
//
// 如果某个 idx 的 npairs > maxPairs_perwarp，下面代码会 fallback 到你原来的标量路径
// ------------------------------------------------------------
__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_streamed_microkernel(
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
    int Nmax_perwarp,          // 每 warp 预留的最大列长
    int maxPairs_perwarp       // 每 warp 预留的最大 pair 数
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

    if (N > Nmax_perwarp) return;

    // --------------------------------------------------------
    // shared memory layout per warp
    //
    // 每个 warp 需要：
    //   sRhs         : 25
    //   sYall        : Nmax_perwarp * 25
    //   Ablocks_pack : maxPairs_perwarp * 25
    //   Yblocks_pack : maxPairs_perwarp * 25
    //   A_panel      : maxPairs_perwarp * 25
    //   B_panel      : maxPairs_perwarp * 25
    //   Csum         : 25
    //
    // 总计 doubles/warp:
    //   25 * (Nmax_perwarp + 4*maxPairs_perwarp + 2)
    // --------------------------------------------------------
    extern __shared__ VALUE_TYPE sh[];

    VALUE_TYPE* warp_base = sh
        + warp * ( BS2 * (Nmax_perwarp + 4 * maxPairs_perwarp + 2) );

    VALUE_TYPE* sRhs         = warp_base;                              // [25]
    VALUE_TYPE* sYall        = sRhs         + BS2;                     // [Nmax][25]
    VALUE_TYPE* Ablocks_pack = sYall        + Nmax_perwarp * BS2;      // [maxPairs][25]
    VALUE_TYPE* Yblocks_pack = Ablocks_pack + maxPairs_perwarp * BS2;  // [maxPairs][25]
    VALUE_TYPE* A_panel      = Yblocks_pack + maxPairs_perwarp * BS2;  // [5][5*maxPairs]
    VALUE_TYPE* B_panel      = A_panel      + maxPairs_perwarp * BS2;  // [5*maxPairs][5]
    VALUE_TYPE* Csum         = B_panel      + maxPairs_perwarp * BS2;  // [25]

    // --------------------------------------------------------
    // 主循环：逐个 idx 计算 Y(idx)
    // --------------------------------------------------------
    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];

        // rhs = (t==j ? I : 0)
        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }
        __syncwarp();

        int g0   = start + idx;
        int qbeg = PairPtr[g0];
        int qend = PairPtr[g0 + 1];
        int npairs = qend - qbeg;

        // ----------------------------------------------------
        // 路径 1：如果 pair 数不大，走新的 streamed microkernel
        // ----------------------------------------------------
        if (npairs > 0 && npairs <= maxPairs_perwarp) {

            // 先把所有 pair 对应的 A_q / Y_q 收集到 shared
            for (int q = 0; q < npairs; ++q) {
                int qq      = qbeg + q;
                int p_local = PairPLocal[qq];
                int srcA    = PairSrc[qq];

                for (int e = lane; e < BS2; e += 32) {
                    Ablocks_pack[q * BS2 + e] = Abval[srcA * BS2 + e];
                    Yblocks_pack[q * BS2 + e] = sYall[p_local * BS2 + e];
                }
            }
            __syncwarp();

            // 用 panel + streamed microkernel 一次算 Csum = sum_q A_q * Y_q
            microkernel_sum_pairs_streamed(
                lane,
                npairs,
                Ablocks_pack,
                Yblocks_pack,
                A_panel,
                B_panel,
                Csum
            );

            // rhs -= Csum
            if (lane < BS2) {
                sRhs[lane] -= Csum[lane];
            }
            __syncwarp();
        }
        // ----------------------------------------------------
        // 路径 2：fallback，走你原来的标量 pair-loop
        // ----------------------------------------------------
        else if (npairs > 0) {
            VALUE_TYPE* sA = A_panel; // 借一块 shared 临时放 A（这里 A_panel 足够大）

            for (int q = qbeg; q < qend; ++q) {
                int p_local = PairPLocal[q];
                int srcA    = PairSrc[q];

                if (lane < BS2) {
                    sA[lane] = Abval[srcA * BS2 + lane];
                }
                __syncwarp();

                if (lane < BS2) {
                    const VALUE_TYPE* Yp = sYall + p_local * BS2;
                    int rr = lane / BS;
                    int cc = lane - rr * BS;
                    VALUE_TYPE sum = 0;
                    #pragma unroll
                    for (int k = 0; k < BS; ++k) {
                        sum += sA[rr * BS + k] * Yp[k * BS + cc];
                    }
                    sRhs[lane] -= sum;
                }
                __syncwarp();
            }
        }

        // ----------------------------------------------------
        // Y = DinvL[t] * rhs
        // ----------------------------------------------------
        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            int rr = lane / BS;
            int cc = lane - rr * BS;

            VALUE_TYPE acc = 0;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                acc += inv[rr * BS + k] * sRhs[k * BS + cc];
            }

            // 写到本列缓存
            sYall[idx * BS2 + lane] = acc;
        }
        __syncwarp();
    }

    // --------------------------------------------------------
    // 最后一次性写回 global
    // --------------------------------------------------------
    for (int idx = 0; idx < N; ++idx) {
        if (lane < BS2) {
            Mval[(start + idx) * BS2 + lane] = sYall[idx * BS2 + lane];
        }
        __syncwarp();
    }
}


// 假设：
//   BS  = 5
//   BS2 = 25
//   VALUE_TYPE = double   （因为 mma_m8n8k4_f64 走的是 f64 Tensor Core 路径）
//
// 说明：
// 1. 这里去掉了 sA / sYp 的 shared memory 中转
// 2. 每个 warp 内：lane 0..24 分别持有 tile(5x5) 的一个元素，布局为 row-major：lane = row*5 + col
// 3. tc_update_rhs_5x5_k4_from_regs() 通过 __shfl_sync 从对应 owner lane 取 A/Yp 元素
// 4. add_tail_k4_outer_product_from_regs() 同理
//
// 你只需要保证：__CUDA_ARCH__ >= 800 且 VALUE_TYPE == double 时走 tensor 路径。


__device__ __forceinline__
double load_fragA_from_sA_5x5_first4cols_pack(const double* sA, int lane)
{
    int row = lane >> 2;   // 0..7
    int col = lane &  3;   // 0..3

    if (row < 5) {
        return sA[row * 5 + col];
    } else {
        return 0.0;
    }
}

// -----------------------------
// B frag from packed shared 4x8
// 直接按 lane 读取，无需 row*5+col
// -----------------------------
__device__ __forceinline__
double load_fragB_from_sYp_packed_4x8(const double* sYpPack, int lane)
{
    return sYpPack[lane];   // lane 0..31 <=> packed 4x8
}

__device__ __forceinline__
void load_acc_from_sRhs_5x5_pack(const double* sRhs, int lane, double acc[2])
{
    int groupID = lane >> 2;           // 0..7 -> row
    int threadID_in_group = lane & 3;  // 0..3
    int row = groupID;
    int c0 = threadID_in_group * 2 + 0;
    int c1 = threadID_in_group * 2 + 1;

    acc[0] = (row < 5 && c0 < 5) ? sRhs[row * 5 + c0] : 0.0;
    acc[1] = (row < 5 && c1 < 5) ? sRhs[row * 5 + c1] : 0.0;
}

__device__ __forceinline__
void store_acc_to_sRhs_5x5_pack(double* sRhs, int lane, const double acc[2])
{
    int groupID = lane >> 2;
    int threadID_in_group = lane & 3;
    int row = groupID;
    int c0 = threadID_in_group * 2 + 0;
    int c1 = threadID_in_group * 2 + 1;

    if (row < 5 && c0 < 5) sRhs[row * 5 + c0] = acc[0];
    if (row < 5 && c1 < 5) sRhs[row * 5 + c1] = acc[1];
}

// sRhs -= A_use(5x4) * Yp_use(4x5)
// A from shared 5x5, B from packed shared 4x8
__device__ __forceinline__
void tc_update_rhs_5x5_k4_sA_shm_sYpPacked(double* sRhs,
                                           const double* sA,
                                           const double* sYpPack)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    int lane = threadIdx.x & 31;

    double fragA = load_fragA_from_sA_5x5_first4cols_pack(sA, lane);
    double fragB = load_fragB_from_sYp_packed_4x8(sYpPack, lane);

    double acc[2];
    load_acc_from_sRhs_5x5_pack(sRhs, lane, acc);

    fragA = -fragA;
    mma_m8n8k4_f64(acc, fragA, fragB);

    store_acc_to_sRhs_5x5_pack(sRhs, lane, acc);
#endif
}

// tail: sRhs -= A(:,4) * Yp(4,:)
// Yp 的第5行单独放在 shared 的 sYtail[5]
__device__ __forceinline__
void add_tail_k4_outer_product_sA_shm_sYtail_shm(double* sRhs,
                                                 const double* sA,
                                                 const double* sYtail)
{
    int lane = threadIdx.x & 31;
    if (lane < 25) {
        int rr = lane / 5;
        int cc = lane % 5;

        double a = sA[rr * 5 + 4];
        double b = sYtail[cc];

        sRhs[rr * 5 + cc] -= a * b;
    }
}

// 将 Yp(5x5,row-major) 打包到：
// 1) sYpPack[32]  : 前4行 -> 4x8
//      col 0..4 有效，col 5..7 补0
// 2) sYtail[5]    : 第5行，给 tail 用
__device__ __forceinline__
void pack_Yp_5x5_to_shared_4x8_and_tail(const VALUE_TYPE* Yp_v,
                                        VALUE_TYPE* sYpPack,
                                        VALUE_TYPE* sYtail,
                                        int lane)
{
    if (lane < 32) {
        int row = lane & 3;   // 0..3
        int col = lane >> 2;  // 0..7
        if (col < 5) {
            sYpPack[lane] = load_double_from_global_cs(Yp_v + row * 5 + col);
        } else {
            sYpPack[lane] = VALUE_TYPE(0.0);
        }
    }

    if (lane < 5) {
        sYtail[lane] = load_double_from_global_cs(Yp_v + 4 * 5 + lane);
    }
}

__global__ void isai_lower_bsr5_warpcol_with_pairs_kernel_tensor_sYp_packed(
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
    const VALUE_TYPE* __restrict__ DinvL
)
{
#if !defined(USE_DOUBLE)
    return;
#else
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;
    int j = blockIdx.x * warpsPerBlock + warp;
    if (j >= nb) return;

    int start = load_int_from_global_cs(Mbrow + j);
    int end   = load_int_from_global_cs(Mbrow + j + 1);
    int N = end - start;
    if (N <= 0) return;

    extern __shared__ VALUE_TYPE sh[];

    // 每个 warp:
    // sRhs   : 25
    // sA     : 25
    // sYpPack: 32   (4x8)
    // sYtail : 5
    // 合计   : 87 doubles
    const int OFF_RHS   = 0;
    const int OFF_A     = OFF_RHS   + BS2;   // 25
    const int OFF_YPACK = OFF_A     + BS2;   // 50
    const int OFF_YTAIL = OFF_YPACK + 32;    // 82
    const int STRIDE    = OFF_YTAIL + 5;     // 87

    VALUE_TYPE* base    = sh + warp * STRIDE;
    VALUE_TYPE* sRhs    = base + OFF_RHS;
    VALUE_TYPE* sA      = base + OFF_A;
    VALUE_TYPE* sYpPack = base + OFF_YPACK;
    VALUE_TYPE* sYtail  = base + OFF_YTAIL;

    for (int idx = 0; idx < N; ++idx) {
        int t = load_int_from_global_cs(Mcol + start + idx);

        // rhs = (t==j ? I : 0)
        if (lane < BS2) {
            int r = lane / BS;
            int c = lane - r * BS;
            sRhs[lane] = (t == j && r == c) ? VALUE_TYPE(1.0) : VALUE_TYPE(0.0);
        }
        __syncwarp();

        int g0   = start + idx;
        int qbeg = load_int_from_global_cs(PairPtr + g0);
        int qend = load_int_from_global_cs(PairPtr + g0 + 1);

        for (int q = qbeg; q < qend; ++q) {
            int p_local = load_int_from_global_cs(PairPLocal + q);
            int srcA    = load_int_from_global_cs(PairSrc + q);

            const VALUE_TYPE* Ablk_v = Abval + srcA * BS2;
            const VALUE_TYPE* Yp_v   = Mval  + (start + p_local) * BS2;

            // sA 仍按原始 5x5 row-major 放 shared
            if (lane < BS2) {
                sA[lane] = load_double_from_global_cs(Ablk_v + lane);
            }

            // sYp 改成 packed 4x8 + tail row
            pack_Yp_5x5_to_shared_4x8_and_tail(Yp_v, sYpPack, sYtail, lane);
            __syncwarp();

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
            const double* sA_d      = reinterpret_cast<const double*>(sA);
            const double* sYpPack_d = reinterpret_cast<const double*>(sYpPack);
            const double* sYtail_d  = reinterpret_cast<const double*>(sYtail);

            tc_update_rhs_5x5_k4_sA_shm_sYpPacked(reinterpret_cast<double*>(sRhs),
                                                  sA_d, sYpPack_d);
            __syncwarp();

            add_tail_k4_outer_product_sA_shm_sYtail_shm(reinterpret_cast<double*>(sRhs),
                                                        sA_d, sYtail_d);
            __syncwarp();
#else
            // fallback 标量版
            if (lane < BS2) {
                int rr = lane / BS;
                int cc = lane - rr * BS;
                VALUE_TYPE sum = VALUE_TYPE(0.0);

#pragma unroll
                for (int kk = 0; kk < 4; ++kk) {
                    sum += sA[rr * BS + kk] * sYpPack[(cc << 2) + kk];
                }
                sum += sA[rr * BS + 4] * sYtail[cc];

                sRhs[lane] -= sum;
            }
            __syncwarp();
#endif
        }

        // Y = DinvL[t] * rhs
        if (lane < BS2) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            int rr = lane / BS;
            int cc = lane - rr * BS;

            VALUE_TYPE acc = VALUE_TYPE(0.0);
#pragma unroll
            for (int kk = 0; kk < BS; ++kk) {
                acc += load_double_from_global_cs(inv + rr * BS + kk) *
                       sRhs[kk * BS + cc];
            }

            Mval[(start + idx) * BS2 + lane] = acc;
        }
        __syncwarp();
    }
#endif
}
