#pragma once
#include "common.h"
#include "config.h"

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#ifndef DBG_CUDA
#define DBG_CUDA(call) do{auto _e=(call); if(_e!=cudaSuccess){ \
  std::cerr<<"CUDA "<<cudaGetErrorString(_e)<<" @"<<__FILE__<<":"<<__LINE__<<"\n"; std::exit(1);} }while(0)
#endif

// 打印一个 5x5 块（行主序）
inline void print_block5(const VALUE_TYPE* blk) {
    std::cout.setf(std::ios::fixed);
    std::cout<<std::setprecision(6);
    for (int r=0; r<BS; ++r) {
        std::cout<<"    ";
        for (int c=0; c<BS; ++c) {
            std::cout<<std::setw(11)<<blk[r*BS + c]<<" ";
        }
        std::cout<<"\n";
    }
}

// 从设备端拷回并打印：RAW（LU打包）、L视图（unit diag）、U视图
inline void dump_bsrilu02_bsr(const char* title,
                              int nb, int nnzb,
                              const int* d_bsrRow,
                              const int* d_bsrCol,
                              const VALUE_TYPE* d_bsrVal)
{
    std::vector<int> hRow(nb+1), hCol(nnzb);
    std::vector<VALUE_TYPE> hVal(nnzb*BS2);
    DBG_CUDA(cudaMemcpy(hRow.data(), d_bsrRow, (nb+1)*sizeof(int), cudaMemcpyDeviceToHost));
    DBG_CUDA(cudaMemcpy(hCol.data(), d_bsrCol, nnzb*sizeof(int),   cudaMemcpyDeviceToHost));
    DBG_CUDA(cudaMemcpy(hVal.data(), d_bsrVal, nnzb*BS2*sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

    std::cout << "\n==== " << title << " ====\n";
    std::cout << "nb(block rows/cols) = " << nb
              << ", nnzb(blocks) = " << nnzb
              << ", BS = " << BS << "\n";

    // 1) RAW（LU打包）—— 这是 d_bsrVal 里真实存的内容
    std::cout << "\n-- RAW blocks (LU packed in one BSR) --\n";
    for (int br=0; br<nb; ++br) {
        for (int p=hRow[br]; p<hRow[br+1]; ++p) {
            int bc = hCol[p];
            const VALUE_TYPE* blk = &hVal[p*BS2];
            std::cout << "  block ("<<br<<","<<bc<<")";
            if (br==bc) std::cout<<"  [diag: this is U_kk]";
            std::cout << ":\n";
            print_block5(blk);
        }
    }

    // 2) L 视图（单位对角）—— 显示下三角块，且对角显示为 I5
    std::cout << "\n-- L view (unit diagonal, shown as I5) --\n";
    for (int br=0; br<nb; ++br) {
        // 先打印对角 I5
        std::cout << "  block ("<<br<<","<<br<<") [L_kk = I5]:\n";
        VALUE_TYPE I5[BS2] = {0};
        for (int d=0; d<BS; ++d) I5[d*BS+d]=static_cast<VALUE_TYPE>(1.0);
        print_block5(I5);

        // 再打印严格下三角
        for (int p=hRow[br]; p<hRow[br+1]; ++p) {
            int bc = hCol[p];
            if (br <= bc) continue; // 严格下
            const VALUE_TYPE* blk = &hVal[p*BS2];
            std::cout << "  block ("<<br<<","<<bc<<"):\n";
            print_block5(blk);
        }
    }

    // 3) U 视图（含对角 U_kk）—— 显示上三角（含对角）
    std::cout << "\n-- U view (includes diagonal U_kk) --\n";
    for (int br=0; br<nb; ++br) {
        for (int p=hRow[br]; p<hRow[br+1]; ++p) {
            int bc = hCol[p];
            if (br > bc) continue; // 上三角 含对角
            const VALUE_TYPE* blk = &hVal[p*BS2];
            std::cout << "  block ("<<br<<","<<bc<<")";
            if (br==bc) std::cout<<"  [U_kk]";
            std::cout << ":\n";
            print_block5(blk);
        }
    }
    std::cout.flush();
}

void debug_check_MU_diag(
    int nb, int nnzb,                       // BSR 的块维/块非零
    const int* d_bsrRow, const int* d_bsrCol, const VALUE_TYPE* d_bsrVal, // RAW(LU)
    const int* d_MbrowU, const int* d_McolU, const VALUE_TYPE* d_MvalU)   // BCSC(U)
{
    // 拷回 host
    std::vector<int> hBr(nb+1), hBc(nnzb);
    std::vector<VALUE_TYPE> hBv(nnzb*BS2);
    CHECK_CUDA(cudaMemcpy(hBr.data(), d_bsrRow, (nb+1)*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hBc.data(), d_bsrCol, nnzb*sizeof(int),   cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hBv.data(), d_bsrVal, nnzb*BS2*sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

    // M_U 的 BCSC
    std::vector<int> hMb(nb+1), hMc; hMc.resize(0);
    CHECK_CUDA(cudaMemcpy(hMb.data(), d_MbrowU, (nb+1)*sizeof(int), cudaMemcpyDeviceToHost));
    hMc.resize(hMb.back());
    std::vector<VALUE_TYPE> hMv(hMb.back()*BS2);
    CHECK_CUDA(cudaMemcpy(hMc.data(), d_McolU, hMc.size()*sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hMv.data(), d_MvalU, hMv.size()*sizeof(VALUE_TYPE), cudaMemcpyDeviceToHost));

    auto find_block = [&](int br, int bc)->const VALUE_TYPE*{
        for (int p=hBr[br]; p<hBr[br+1]; ++p) if (hBc[p]==bc) return &hBv[p*BS2];
        return nullptr;
    };
    auto find_MU_diag = [&](int j)->const VALUE_TYPE*{
        for (int p=hMb[j]; p<hMb[j+1]; ++p) if (hMc[p]==j) return &hMv[p*BS2];
        return nullptr;
    };

    std::cout << "\n[DBG] Check U_kk * M_U(j,j) vs I (Frobenius norm)\n";
    for (int j=0; j<nb; ++j) {
        const VALUE_TYPE* Ukk = find_block(j,j);
        const VALUE_TYPE* Mjj = find_MU_diag(j);
        if (!Ukk || !Mjj) { std::cout << "  j="<<j<<"  (missing block)\n"; continue; }

        // C = Ukk * Mjj
        VALUE_TYPE C[BS2]={0};
        for (int i=0;i<BS;++i)
            for (int k=0;k<BS;++k)
                for (int l=0;l<BS;++l)
                    C[i*BS+l] += Ukk[i*BS+k]*Mjj[k*BS+l];

        // C - I 的 Frobenius 范数
        VALUE_TYPE fn=static_cast<VALUE_TYPE>(0.0);
        for (int i=0;i<BS;++i)
            for (int l=0;l<BS;++l){
                VALUE_TYPE v = C[i*BS+l] - static_cast<VALUE_TYPE>(i==l?1.0:0.0);
                fn += v*v;
            }
        std::cout << "  j="<<j<<"  ||U_kk*Mjj - I||_F = " << std::sqrt(fn) << "\n";
    }
}

void inspect_bcsc_column_stats(const char* name,
                               int nb,
                               const std::vector<int>& Mbrow,
                               const std::vector<int>& Mcol,
                               int blockDimLimit = 128,
                               int topK = 10)
{
    auto bin = [](int N)->int{
        if (N<=8) return 0; if (N<=16) return 1; if (N<=32) return 2;
        if (N<=64) return 3; if (N<=128) return 4; if (N<=256) return 5;
        return 6;
    };
    const char* binName[7] = {"1-8","9-16","17-32","33-64","65-128","129-256",">256"};

    std::vector<int> hist(7,0);
    long long sumN = 0;
    int maxN = 0, minN = (nb>0? INT_MAX:0), cntOver = 0, cntNoDiag = 0, cntUnsorted = 0;

    struct ColInfo { int j, N, diagPos; bool diagLast; };
    std::vector<ColInfo> all;

    for (int j=0; j<nb; ++j) {
        int start = Mbrow[j], end = Mbrow[j+1];
        int N = end - start;
        minN = std::min(minN, N);
        maxN = std::max(maxN, N);
        sumN += N;
        hist[bin(N)]++;

        // 升序性 & 对角位置
        bool sorted = true;
        int diagPos = -1;
        int prev = (N>0 ? Mcol[start] : -1);
        for (int k=0; k<N; ++k) {
            int r = Mcol[start + k];
            if (k>0 && r < prev) sorted = false;
            if (r == j && diagPos < 0) diagPos = k;
            prev = r;
        }
        if (!sorted) cntUnsorted++;
        if (diagPos < 0) cntNoDiag++;

        if (N > blockDimLimit) cntOver++;

        all.push_back({j, N, diagPos, (diagPos == N-1 && N>0)});
    }

    // 输出总体统计
    std::cout << "\n[BCSC stats] " << name << "\n";
    std::cout << "  nb=" << nb
              << "  avgN=" << std::fixed << std::setprecision(2)
              << (nb? static_cast<VALUE_TYPE>(sumN)/static_cast<VALUE_TYPE>(nb) : static_cast<VALUE_TYPE>(0.0))
              << "  minN=" << minN
              << "  maxN=" << maxN
              << "  count(N>" << blockDimLimit << ")=" << cntOver
              << "  noDiag=" << cntNoDiag
              << "  unsortedCols=" << cntUnsorted << "\n";
    std::cout << "  histogram(N per column): ";
    for (int b=0;b<7;++b){
        std::cout << binName[b] << ":" << hist[b] << (b==6?'\n':' ');
    }

    // 列出 N 最大的 topK 列
    std::sort(all.begin(), all.end(), [](const ColInfo& a, const ColInfo& b){
        if (a.N!=b.N) return a.N>b.N;
        return a.j<b.j;
    });
    std::cout << "  Top-" << topK << " widest columns:\n";
    for (int t=0; t<std::min(topK, (int)all.size()); ++t) {
        auto c = all[t];
        std::cout << "    j=" << c.j
                  << "  N=" << c.N
                  << "  diagPos=" << c.diagPos
                  << "  diagLast=" << (c.diagLast?"Y":"N")
                  << (c.N>blockDimLimit?"  [>blockDim]":"")
                  << (c.diagPos<0?"  [NO_DIAG]":"")
                  << "\n";
    }
}
