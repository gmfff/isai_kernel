#include "common.h"
#include <vector>
#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include "config.h"
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
        for (int e = lane; e < BS2; e += 32) {
            int r = e / BS;
            int c = e - r * BS;
            sRhs[e] = (t == j && r == c) ? 1.0 : 0.0;
        }
        __syncwarp();

        // rhs -= sum_{p>idx} U(t, colp) * Y_p
        for (int p = N - 1; p > idx; --p) {
            int colp = Mcol[start + p];

            if (lane == 0) {
                bsr_lookup5_full(t, colp, Abrow, Abcol, Abval, sA);
            }
            for (int e = lane; e < BS2; e += 32) {
                sYp[e] = Mval[(start + p) * BS2 + e];
            }
            __syncwarp();

            for (int e = lane; e < BS2; e += 32) {
                int i = e / BS;
                int j2 = e - i * BS;
                VALUE_TYPE sum = 0.0;
                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    sum += sA[i * BS + k] * sYp[k * BS + j2];
                }
                sRhs[e] -= sum;
            }
            __syncwarp();
        }

        // Y = DinvU[t] * rhs
        for (int e = lane; e < BS2; e += 32) {
            const VALUE_TYPE* inv = DinvU + t * BS2;
            int i = e / BS;
            int j2 = e - i * BS;
            VALUE_TYPE sum = 0.0;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                sum += inv[i * BS + k] * sRhs[k * BS + j2];
            }
            Mval[(start + idx) * BS2 + e] = sum;
        }
        __syncwarp();
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
        for (int e = lane; e < BS2; e += 32) {
            int r = e / BS;
            int c = e - r * BS;
            sRhs[e] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }
        __syncwarp();

        for (int q = qbeg; q < qend; ++q) {
            int p_local = PairPLocal[q];  // 0..idx-1
            int srcA    = PairSrc[q];

            // 读 A(t,p) -> sA
            for (int e = lane; e < BS2; e += 32) {
                sA[e] = Abval[srcA * BS2 + e];
            }
            __syncwarp();
            // 读 Yp：从 shared 缓存读，不再从 global 读
            for (int e = lane; e < BS2; e += 32) {
                // p_local 一定 < idx，因此一定已经写入 sYall
                // sYall 的布局：连续存 idx=0..N-1 的块
                // 每块 25 个
                // 取出第 p_local 块
                const VALUE_TYPE* Yp = sYall + p_local * BS2;
                // 为了保持你原逻辑，我们用 sRhs -= sA*Yp 直接算，不必再拷到 sYp 临时数组
                int rr = e / BS;
                int cc = e - rr * BS;
                VALUE_TYPE sum = 0;
                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    sum += sA[rr * BS + k] * Yp[k * BS + cc];
                }
                sRhs[e] -= sum;
            }
            __syncwarp();
        }

        // Y = DinvL[t] * rhs（标量 5×5）
        for (int e = lane; e < BS2; e += 32) {
            const VALUE_TYPE* inv = DinvL + t * BS2;
            int rr = e / BS;
            int cc = e - rr * BS;
            VALUE_TYPE acc = 0;
            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                acc += inv[rr * BS + k] * sRhs[k * BS + cc];
            }
            // 先写到 shared 缓存（本列内后续会用到）
            sYall[idx * BS2 + e] = acc;
        }
        __syncwarp();
    }

    // 最后一次性把本列所有块写回 global（减少 global 往返）
    for (int idx = 0; idx < N; ++idx) {
        for (int e = lane; e < BS2; e += 32) {
            Mval[(start + idx) * BS2 + e] = sYall[idx * BS2 + e];
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

    if (row < BS && col < BS) {
        return sA[row * BS + col];
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

    if (row < BS && col < BS) {
        return Yp[row * BS + col];
    } else {
        return 0.0;
    }
}

__device__ __forceinline__
double load_fragA_from_sA_7x7_cols4to7(const double* sA, int lane)
{
    int row = lane >> 2;       // 0..7
    int col = (lane & 3) + 4;  // 4..7

    if (BS == 7 && row < BS && col < BS) {
        return sA[row * BS + col];
    }
    return 0.0;
}

__device__ __forceinline__
double load_fragB_from_Yp_7x7_rows4to7(const double* Yp, int lane)
{
    int row = (lane & 3) + 4;  // 4..7
    int col = lane >> 2;       // 0..7

    if (BS == 7 && row < BS && col < BS) {
        return Yp[row * BS + col];
    }
    return 0.0;
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

    acc[0] = (row < BS && c0 < BS) ? sRhs[row * BS + c0] : 0.0;
    acc[1] = (row < BS && c1 < BS) ? sRhs[row * BS + c1] : 0.0;
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

    if (row < BS && c0 < BS) sRhs[row * BS + c0] = acc[0];
    if (row < BS && c1 < BS) sRhs[row * BS + c1] = acc[1];
}

// A_embed: 8x4
// 从 inv(5x5,row-major) 里取前4列，组成 5x4；再补到 8x4
__device__ __forceinline__
double load_fragA_from_inv_5x5_first4cols(const double* inv, int lane)
{
    int row = lane >> 2;   // 0..7
    int col = lane &  3;   // 0..3

    if (row < BS && col < BS) return inv[row * BS + col];
    return 0.0;
}

// B_embed: 4x8
// 从 rhs(5x5,row-major) 里取前4行，组成 4x5；再补到 4x8
__device__ __forceinline__
double load_fragB_from_rhs_5x5_first4rows(const double* rhs, int lane)
{
    int row = lane &  3;   // 0..3
    int col = lane >> 2;   // 0..7

    if (row < BS && col < BS) return rhs[row * BS + col];
    return 0.0;
}

__device__ __forceinline__
double load_fragA_from_inv_7x7_cols4to7(const double* inv, int lane)
{
    int row = lane >> 2;
    int col = (lane & 3) + 4;

    if (BS == 7 && row < BS && col < BS) return inv[row * BS + col];
    return 0.0;
}

__device__ __forceinline__
double load_fragB_from_rhs_7x7_rows4to7(const double* rhs, int lane)
{
    int row = (lane & 3) + 4;
    int col = lane >> 2;

    if (BS == 7 && row < BS && col < BS) return rhs[row * BS + col];
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

    if (row < BS && c0 < BS) Y[row * BS + c0] = acc[0];
    if (row < BS && c1 < BS) Y[row * BS + c1] = acc[1];
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

    if (BS == 7) {
        fragA = load_fragA_from_inv_7x7_cols4to7(inv, lane);
        fragB = load_fragB_from_rhs_7x7_rows4to7(rhs, lane);
        mma_m8n8k4_f64(acc, fragA, fragB);
    }

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

    if (BS == 7) return;

    for (int e = lane; e < BS2; e += 32) {
        int rr = e / BS;
        int cc = e % BS;
        double sum = 0.0;
        #pragma unroll
        for (int k = 4; k < BS; ++k) {
            sum += inv[rr * BS + k] * rhs[k * BS + cc];
        }
        Y[rr * BS + cc] += sum;
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
    return (row < BS && col < BS) ? load_double_cs_bsr5(Ablk + row * BS + col) : 0.0;
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

    if (BS == 7) {
        fragA = -load_fragA_from_sA_7x7_cols4to7(sA, lane);
        fragB = load_fragB_from_Yp_7x7_rows4to7(Yp, lane);
        mma_m8n8k4_f64(acc, fragA, fragB);
    }

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

    if (BS == 7) {
        int row = lane >> 2;
        int col = (lane & 3) + 4;
        fragA = (row < BS && col < BS)
            ? -load_double_cs_bsr5(Ablk + row * BS + col)
            : 0.0;
        fragB = load_fragB_from_Yp_7x7_rows4to7(Yp, lane);
        mma_m8n8k4_f64(acc, fragA, fragB);
    }

    store_acc_to_sRhs_5x5(sRhs, lane, acc);
#endif
}

__device__ __forceinline__
void add_tail_k4_outer_product(double* sRhs, const double* sA, const double* Yp)
{
    int lane = threadIdx.x & 31;

    if (BS == 7) return;

    for (int e = lane; e < BS2; e += 32) {
        int rr = e / BS;
        int cc = e % BS;

        //double a = sA[rr * 5 + 4];   // A 的第 5 列
        //double b = Yp[4 * 5 + cc];   // B 的第 5 行
        double sum = 0.0;
        #pragma unroll
        for (int k = 4; k < BS; ++k) {
            sum += sA[rr * BS + k] * Yp[k * BS + cc];
        }
        sRhs[rr * BS + cc] -= sum;
    }
}

__device__ __forceinline__
void add_tail_k4_outer_product_Aglobal_Yshared(
    double* sRhs,
    const double* Ablk,
    const double* Yp)
{
    int lane = threadIdx.x & 31;

    if (BS == 7) return;

    for (int e = lane; e < BS2; e += 32) {
        int rr = e / BS;
        int cc = e % BS;
        double sum = 0.0;
        #pragma unroll
        for (int k = 4; k < BS; ++k) {
            double a = load_double_cs_bsr5(Ablk + rr * BS + k);
            sum += a * Yp[k * BS + cc];
        }
        sRhs[rr * BS + cc] -= sum;
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
        for (int e = lane; e < BS2; e += 32) {
            int r = e / BS;
            int c = e - r * BS;
            sRhs[e] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }
        __syncwarp();

        for (int q = qbeg; q < qend; ++q) {
            int p_local = PairPLocal[q];  // 0..idx-1
            int srcA    = PairSrc[q];

            for (int e = lane; e < BS2; e += 32) {
                sA[e] = Abval[srcA * BS2 + e];
            }
            __syncwarp();
            {
                const VALUE_TYPE* Yp = sYall + p_local * BS2;

            #if __CUDA_ARCH__ >= 800
                tc_update_rhs_5x5_k4((double*)sRhs, (const double*)sA, (const double*)Yp);

                add_tail_k4_outer_product(sRhs, sA, Yp);
            #else
                // fallback: 原始标量版本
                for (int e = lane; e < BS2; e += 32) {
                    int rr = e / BS;
                    int cc = e - rr * BS;
                    VALUE_TYPE sum = 0;
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {   // 这里只算前4列/前4行
                        sum += sA[rr * BS + k] * Yp[k * BS + cc];
                    }
                    sRhs[e] -= sum;
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
            for (int e = lane; e < BS2; e += 32) {
                int rr = e / BS;
                int cc = e - rr * BS;
                VALUE_TYPE acc = 0;

                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    acc += inv[rr * BS + k] * sRhs[k * BS + cc];
                }

                Ycur[e] = acc;
            }
        #endif
        }
        __syncwarp();
    }

    for (int idx = 0; idx < N; ++idx) {
        for (int e = lane; e < BS2; e += 32) {
            Mval[(start + idx) * BS2 + e] = sYall[idx * BS2 + e];
        }
        __syncwarp();
    }
}

__global__ void isai_lower_bsr4_warpcol_with_pairs_kernel_cachedY_tensor_notail(
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
    int Nmax_perwarp)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;

    int j = blockIdx.x * warpsPerBlock + warp;
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N <= 0 || N > Nmax_perwarp) return;

    extern __shared__ VALUE_TYPE sh[];
    VALUE_TYPE* warp_base = sh + warp * ((2 * BS2) + Nmax_perwarp * BS2);

    VALUE_TYPE* sRhs   = warp_base + 0;
    VALUE_TYPE* sA     = warp_base + BS2;
    VALUE_TYPE* sYall  = warp_base + 2 * BS2;

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];
        int g0   = start + idx;
        int qbeg = PairPtr[g0];
        int qend = PairPtr[g0 + 1];
        VALUE_TYPE* Ycur = sYall + idx * BS2;

        for (int e = lane; e < BS2; e += 32) {
            int r = e / BS;
            int c = e - r * BS;
            sRhs[e] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }
        __syncwarp();

        for (int q = qbeg; q < qend; ++q) {
            int p_local = PairPLocal[q];
            int srcA    = PairSrc[q];

            for (int e = lane; e < BS2; e += 32) {
                sA[e] = Abval[srcA * BS2 + e];
            }

            const VALUE_TYPE* Yp = sYall + p_local * BS2;

        #if __CUDA_ARCH__ >= 800
            tc_update_rhs_5x5_k4((double*)sRhs, (const double*)sA, (const double*)Yp);
        #else
            for (int e = lane; e < BS2; e += 32) {
                int rr = e / BS;
                int cc = e - rr * BS;
                VALUE_TYPE sum = 0;
                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    sum += sA[rr * BS + k] * Yp[k * BS + cc];
                }
                sRhs[e] -= sum;
            }
        #endif
            __syncwarp();
        }

        const VALUE_TYPE* inv = DinvL + t * BS2;

    #if __CUDA_ARCH__ >= 800
        tc_gemm_Dinv_rhs_5x5_k4_store(
            (double*)Ycur,
            (const double*)inv,
            (const double*)sRhs
        );
    #else
        for (int e = lane; e < BS2; e += 32) {
            int rr = e / BS;
            int cc = e - rr * BS;
            VALUE_TYPE acc = 0;

            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                acc += inv[rr * BS + k] * sRhs[k * BS + cc];
            }

            Ycur[e] = acc;
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

__global__ void isai_lower_bsr7_warpcol_with_pairs_kernel_cachedY_tensor_twomma(
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
    int Nmax_perwarp)
{
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warpsPerBlock = blockDim.x >> 5;

    int j = blockIdx.x * warpsPerBlock + warp;
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j + 1];
    int N     = end - start;
    if (N <= 0 || N > Nmax_perwarp) return;

    extern __shared__ VALUE_TYPE sh[];
    VALUE_TYPE* warp_base = sh + warp * ((2 * BS2) + Nmax_perwarp * BS2);

    VALUE_TYPE* sRhs  = warp_base + 0;
    VALUE_TYPE* sA    = warp_base + BS2;
    VALUE_TYPE* sYall = warp_base + 2 * BS2;

    for (int idx = 0; idx < N; ++idx) {
        int t = Mcol[start + idx];
        int g0   = start + idx;
        int qbeg = PairPtr[g0];
        int qend = PairPtr[g0 + 1];
        VALUE_TYPE* Ycur = sYall + idx * BS2;

        for (int e = lane; e < BS2; e += 32) {
            int r = e / BS;
            int c = e - r * BS;
            sRhs[e] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }
        __syncwarp();

        for (int q = qbeg; q < qend; ++q) {
            int p_local = PairPLocal[q];
            int srcA    = PairSrc[q];

            for (int e = lane; e < BS2; e += 32) {
                sA[e] = Abval[srcA * BS2 + e];
            }
            __syncwarp();

            const VALUE_TYPE* Yp = sYall + p_local * BS2;

        #if __CUDA_ARCH__ >= 800
            tc_update_rhs_5x5_k4((double*)sRhs, (const double*)sA, (const double*)Yp);
        #else
            for (int e = lane; e < BS2; e += 32) {
                int rr = e / BS;
                int cc = e - rr * BS;
                VALUE_TYPE sum = 0;
                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    sum += sA[rr * BS + k] * Yp[k * BS + cc];
                }
                sRhs[e] -= sum;
            }
        #endif
            __syncwarp();
        }

        const VALUE_TYPE* inv = DinvL + t * BS2;

    #if __CUDA_ARCH__ >= 800
        tc_gemm_Dinv_rhs_5x5_k4_store(
            (double*)Ycur,
            (const double*)inv,
            (const double*)sRhs
        );
    #else
        for (int e = lane; e < BS2; e += 32) {
            int rr = e / BS;
            int cc = e - rr * BS;
            VALUE_TYPE acc = 0;

            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                acc += inv[rr * BS + k] * sRhs[k * BS + cc];
            }

            Ycur[e] = acc;
        }
    #endif
        __syncwarp();
    }

    for (int idx = 0; idx < N; ++idx) {
        for (int e = lane; e < BS2; e += 32) {
            Mval[(start + idx) * BS2 + e] = sYall[idx * BS2 + e];
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

        for (int e = lane; e < BS2; e += 32) {
            int r = e / BS;
            int c = e - r * BS;
            sRhs[e] = (t == j && r == c) ? (VALUE_TYPE)1 : (VALUE_TYPE)0;
        }
        __syncwarp();

        int g0   = start + idx;
        int qbeg = PairPtr[g0];
        int qend = PairPtr[g0 + 1];

        for (int q = qbeg; q < qend; ++q) {
            int p_local = PairPLocal[q];
            int srcA    = PairSrc[q];

            for (int e = lane; e < BS2; e += 32) {
                sA[e] = Abval[srcA * BS2 + e];
            }
            __syncwarp();

            const VALUE_TYPE* Yp = Mval + (start + p_local) * BS2;

#if __CUDA_ARCH__ >= 800
            tc_update_rhs_5x5_k4((double*)sRhs, (const double*)sA, (const double*)Yp);
            add_tail_k4_outer_product(sRhs, sA, Yp);
#else
            for (int e = lane; e < BS2; e += 32) {
                int rr = e / BS;
                int cc = e - rr * BS;
                VALUE_TYPE sum = 0;
                #pragma unroll
                for (int k = 0; k < BS; ++k) {
                    sum += sA[rr * BS + k] * Yp[k * BS + cc];
                }
                sRhs[e] -= sum;
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
        for (int e = lane; e < BS2; e += 32) {
            int rr = e / BS;
            int cc = e - rr * BS;
            VALUE_TYPE acc = 0;

            #pragma unroll
            for (int k = 0; k < BS; ++k) {
                acc += inv[rr * BS + k] * sRhs[k * BS + cc];
            }

            Ycur[e] = acc;
        }
#endif
        __syncwarp();
    }
}

__device__ __forceinline__
void isai_lower_bsr4_light_globalY_tensor_col_body_notail(
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
void isai_lower_bsr7_light_globalY_tensor_col_body_twomma(
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
    isai_lower_bsr5_light_globalY_tensor_col_body(
        j, nb, Abrow, Abcol, Abval, Mbrow, Mcol,
        PairPtr, PairPLocal, PairSrc, Mval, DinvL, sh, warp);
}

__global__ void isai_lower_bsr4_warpcol_with_pairs_kernel_globalY_tensor_notail(
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
    isai_lower_bsr4_light_globalY_tensor_col_body_notail(
        j, nb, Abrow, Abcol, Abval, Mbrow, Mcol,
        PairPtr, PairPLocal, PairSrc, Mval, DinvL, sh, warp);
}

__global__ void isai_lower_bsr7_warpcol_with_pairs_kernel_globalY_tensor_twomma(
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
    isai_lower_bsr7_light_globalY_tensor_col_body_twomma(
        j, nb, Abrow, Abcol, Abval, Mbrow, Mcol,
        PairPtr, PairPLocal, PairSrc, Mval, DinvL, sh, warp);
}

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
