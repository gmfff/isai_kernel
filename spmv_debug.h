// spmv_debug.h
#pragma once
#include "common.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include "config.h"   // 需要 BS, BS2 与 CHECK_CUDA

namespace spmvdbg {

// ---------- 打印：把 device 向量拷回并按块打印 ----------
inline void dump_device_block_vector5(const char* name,
                                      int nb,
                                      const VALUE_TYPE* d_v,
                                      int max_blocks_to_print = 10)
{
    const int n = nb * BS;
    std::vector<VALUE_TYPE> h_v(n);
    CHECK_CUDA(cudaMemcpy(h_v.data(), d_v, n * sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

    std::cout << "\n" << name << " (n=" << n << ", nb=" << nb << ", BS=" << BS << ")\n";
    int nb_print = std::min(nb, max_blocks_to_print);
    for (int jb = 0; jb < nb_print; ++jb) {
        std::cout << "  block " << jb << " : ";
        for (int r = 0; r < BS; ++r) {
            std::cout << std::setw(12) << std::setprecision(6) << std::fixed
                      << h_v[jb * BS + r] << " ";
        }
        std::cout << "\n";
    }
    if (nb_print < nb) {
        std::cout << "  ... (" << (nb - nb_print) << " more blocks not shown)\n";
    }
    // 2-范数
    VALUE_TYPE nrm = 0.0;
    for (VALUE_TYPE v : h_v) nrm += v * v;
    std::cout << "  ||" << name << "||2 = " << std::setprecision(6) << std::fixed
              << std::sqrt(nrm) << "\n";
}

// ---------- CPU 参考：BCSC(block,5) * x → y_ref（覆盖写，非 +=） ----------
inline void bcsc_matvec_bsr5_cpu(int nb,
                                 const std::vector<int>& Mbrow,
                                 const std::vector<int>& Mcol,
                                 const std::vector<VALUE_TYPE>& Mval,  // nnzb*25, 行主序
                                 const std::vector<VALUE_TYPE>& x,     // nb*5
                                 std::vector<VALUE_TYPE>& y_ref)       // nb*5
{
    y_ref.assign(nb*BS, 0.0);
    for (int j = 0; j < nb; ++j) {
        const VALUE_TYPE* xj = &x[j*BS];
        for (int p = Mbrow[j]; p < Mbrow[j+1]; ++p) {
            int i = Mcol[p];
            const VALUE_TYPE* Bij = &Mval[p*BS2];
            VALUE_TYPE* yi = &y_ref[i*BS];
            for (int r = 0; r < BS; ++r) {
                VALUE_TYPE s = 0.0;
                for (int c = 0; c < BS; ++c)
                    s += Bij[r*BS + c] * xj[c];
                yi[r] += s;
            }
        }
    }
}

// ---------- 串行逐列（无原子）调试 kernel + launcher ----------
/*static __global__ void bcsc_matvec_onecol_noatomic(
    int j, int nb,
    const int* __restrict__ Mbrow,
    const int* __restrict__ Mcol,
    const VALUE_TYPE* __restrict__ Mval,   // nnzb*25
    const VALUE_TYPE* __restrict__ x,      // nb*5
    VALUE_TYPE* __restrict__ y)            // nb*5 (调用前需清零或你自己管理)
{
    int tid = threadIdx.x;
    if (j >= nb) return;
    int start = Mbrow[j], end = Mbrow[j+1];
    int N = end - start;
    if (tid >= N) return;

    int i = Mcol[start + tid];
    const VALUE_TYPE* Bij = Mval + (start + tid) * BS2;
    const VALUE_TYPE* xj  = x + j * BS;

    VALUE_TYPE acc[BS];
    #pragma unroll
    for (int r = 0; r < BS; ++r) acc[r] = 0.0;
    #pragma unroll
    for (int r = 0; r < BS; ++r) {
        VALUE_TYPE s = 0.0;
        #pragma unroll
        for (int c = 0; c < BS; ++c) s += Bij[r*BS + c] * xj[c];
        acc[r] = s;
    }
    VALUE_TYPE* yi = y + i * BS;
    #pragma unroll
    for (int r = 0; r < BS; ++r) yi[r] += acc[r];
}*/
static __global__ void bcsc_matvec_onecol_noatomic(
    int j, int nb,
    const int* __restrict__ Mbrow,
    const int* __restrict__ Mcol,
    const VALUE_TYPE* __restrict__ Mval,   // nnzb * BS*BS
    const VALUE_TYPE* __restrict__ x,      // nb * BS
    VALUE_TYPE* __restrict__ y)            // nb * BS (调用前需清零)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // 修正线程索引
    if (j >= nb) return;

    int start = Mbrow[j];
    int end   = Mbrow[j+1];
    int N = end - start;
    if (tid >= N) return;

    int i = Mcol[start + tid];
    const VALUE_TYPE* Bij = Mval + (start + tid) * BS2;
    const VALUE_TYPE* xj  = x + j * BS;

    VALUE_TYPE acc[BS];
    #pragma unroll
    for (int r = 0; r < BS; ++r) acc[r] = 0.0;

    #pragma unroll
    for (int r = 0; r < BS; ++r)
    {
        VALUE_TYPE s = 0.0;
        #pragma unroll
        for (int c = 0; c < BS; ++c)
            s += Bij[r*BS + c] * xj[c];
        acc[r] = s;
    }

    VALUE_TYPE* yi = y + i * BS;

    // 并行安全累加
    #pragma unroll
    for (int r = 0; r < BS; ++r)
        atomicAdd(&yi[r], acc[r]);
}

inline void launch_bcsc_matvec_add_bsr5_serial_debug(
    int nb,
    const int* d_Mbrow, const int* d_Mcol, const VALUE_TYPE* d_Mval,
    const VALUE_TYPE* d_x, VALUE_TYPE* d_y)
{
    // 逐列发 kernel，列内线程覆盖该列 N（你的 maxN=12，给 32 足够）
    for (int j = 0; j < nb; ++j) {
        int h_start=0, h_end=0;
        CHECK_CUDA(cudaMemcpy(&h_start, d_Mbrow + j,     sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&h_end,   d_Mbrow + j + 1, sizeof(int), cudaMemcpyDeviceToHost));
        int N = h_end - h_start;
        int blk = (N > 0 ? ((N + 31) / 32) * 32 : 32);
        bcsc_matvec_onecol_noatomic<<<1, blk>>>(j, nb, d_Mbrow, d_Mcol, d_Mval, d_x, d_y);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

// ---------- 一键对比：GPU y 与 CPU 参考（自动从 device 拷结构与向量） ----------
inline void compare_gpu_cpu_bcsc_spmv5(const char* tag,
                                       int nb,
                                       const int* d_Mbrow,
                                       const int* d_Mcol,
                                       const VALUE_TYPE* d_Mval,
                                       const VALUE_TYPE* d_x,
                                       const VALUE_TYPE* d_y,
                                       int print_first_blocks = 5)
{
    const int n = nb * BS;

    // 先把 Mbrow 拷回拿到 nnzb
    std::vector<int> h_Mbrow(nb+1);
    CHECK_CUDA(cudaMemcpy(h_Mbrow.data(), d_Mbrow, (nb+1)*sizeof(int), cudaMemcpyDeviceToHost));
    int nnzb = h_Mbrow.back();

    std::vector<int> h_Mcol(nnzb);
    std::vector<VALUE_TYPE> h_Mval(nnzb*BS2), h_x(n), h_y(n), y_ref(n);

    CHECK_CUDA(cudaMemcpy(h_Mcol.data(), d_Mcol, nnzb*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_Mval.data(), d_Mval, nnzb*BS2*sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_x.data(),    d_x,    n*sizeof(VALUE_TYPE),        cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y.data(),    d_y,    n*sizeof(VALUE_TYPE),        cudaMemcpyDeviceToHost));

    // CPU 参考
    bcsc_matvec_bsr5_cpu(nb, h_Mbrow, h_Mcol, h_Mval, h_x, y_ref);

    // 误差
    VALUE_TYPE max_abs = 0.0, nrm_ref = 0.0, nrm_diff = 0.0;
    for (int i=0;i<n;++i){
        VALUE_TYPE diff = std::abs(h_y[i] - y_ref[i]);
        if (diff > max_abs) max_abs = diff;
        nrm_diff += (h_y[i]-y_ref[i])*(h_y[i]-y_ref[i]);
        nrm_ref  += y_ref[i]*y_ref[i];
    }
    nrm_diff = std::sqrt(nrm_diff);
    nrm_ref  = std::sqrt(nrm_ref);

    std::cout << "\n[BCSC SpMV check] " << tag
              << "\n  nnzb=" << nnzb
              << "  ||y_gpu - y_cpu||2=" << std::setprecision(6) << std::fixed << nrm_diff
              << "  max|diff|=" << max_abs
              << "  rel2=" << (nrm_ref>0 ? nrm_diff/nrm_ref : 0.0) << "\n";

    // 打印前几个块（GPU 与 CPU）
    int nb_print = std::min(nb, print_first_blocks);
    for (int jb=0; jb<nb_print; ++jb){
        std::cout << "  block " << jb << " (GPU vs CPU):\n    GPU: ";
        for (int r=0;r<BS;++r)
            std::cout << std::setw(12) << h_y[jb*BS + r] << " ";
        std::cout << "\n    CPU: ";
        for (int r=0;r<BS;++r)
            std::cout << std::setw(12) << y_ref[jb*BS + r] << " ";
        std::cout << "\n";
    }
}

} // namespace spmvdbg
