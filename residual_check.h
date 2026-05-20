// residual_check.h
#pragma once
#include "common.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <limits>
#include "config.h"   // 用到 CHECK_CUDA（若没有可去掉）

// 说明：
// - A 用 CSR（行指针/列索引/数值）存在 host 侧
// - b/x_true 在 host 侧（x_true 可传 null 指针表示没有真解）
// - d_x 是 device 上的解向量；函数内会拷回并计算：
//      rel_error_x = ||x_approx - x_true||2 / ||x_true||2   （若提供 x_true）
//      rel_residual = ||b - A*x_approx||2 / ||b||2
template <typename T>
inline void compute_rel_errors_csr(
    int n,
    const int* csrRowPtrA,        // size n+1 (host)
    const int* csrColIdxA,        // size nnz  (host)
    const T*   csrValA,           // size nnz  (host)
    const T*   b_host,            // size n    (host)
    const T*   x_true,            // size n or nullptr (host)
    const T*   d_x,               // size n    (device)
    const char* tag = "ISAI")
{
    // 1) 从 device 拷回近似解
    std::vector<T> x_approx(n);
#ifdef CHECK_CUDA
    CHECK_CUDA(cudaMemcpy(x_approx.data(), d_x, n * sizeof(T), cudaMemcpyDeviceToHost));
#else
    cudaMemcpy(x_approx.data(), d_x, n * sizeof(T), cudaMemcpyDeviceToHost);
#endif

    // 2) 若有真解，计算解相对误差
    if (x_true) {
        VALUE_TYPE norm_diff_x = 0.0, norm_x_true = 0.0;
        for (int i = 0; i < n; ++i) {
            VALUE_TYPE diff = (VALUE_TYPE)x_approx[i] - (VALUE_TYPE)x_true[i];
            norm_diff_x += diff * diff;
            norm_x_true += (VALUE_TYPE)x_true[i] * (VALUE_TYPE)x_true[i];
        }
        norm_diff_x = std::sqrt(norm_diff_x);
        norm_x_true = std::sqrt(norm_x_true);
        std::cout << "[" << tag << "] rel_error_x = "
                  << (norm_x_true > 0.0 ? norm_diff_x / norm_x_true : 0.0)
                  << "\n";
    }

    // 3) 计算 r = b - A*x_approx 的相对残差（纯 CPU）
    std::vector<VALUE_TYPE> Ax(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int p = csrRowPtrA[i]; p < csrRowPtrA[i+1]; ++p) {
            int j = csrColIdxA[p];
            Ax[i] += (VALUE_TYPE)csrValA[p] * (VALUE_TYPE)x_approx[j];
        }
    }

    VALUE_TYPE norm_r = 0.0, norm_b = 0.0;
    for (int i = 0; i < n; ++i) {
        VALUE_TYPE r = (VALUE_TYPE)b_host[i] - Ax[i];
        norm_r += r * r;
        norm_b += (VALUE_TYPE)b_host[i] * (VALUE_TYPE)b_host[i];
    }
    norm_r = std::sqrt(norm_r);
    norm_b = std::sqrt(norm_b);

    std::cout << "[" << tag << "] rel_residual = "
              << (norm_b > 0.0 ? norm_r / norm_b : std::numeric_limits<VALUE_TYPE>::infinity())
              << "  (||r||2=" << norm_r << ", ||b||2=" << norm_b << ")\n";
}