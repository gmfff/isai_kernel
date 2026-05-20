#pragma once

#include "common.h"

#include <vector>
#include <map>
#include <array>
#include <iostream>
#include "config.h"

static void build_triangular_bcsr_pattern_from_A(
    int nb,
    const int* A_bsrRow,
    const int* A_bsrCol,
    bool lower,
    std::vector<int>& T_bsrRow,
    std::vector<int>& T_bsrCol)
{
    T_bsrRow.assign(nb + 1, 0);
    T_bsrCol.clear();

    for (int r = 0; r < nb; ++r) {
        for (int p = A_bsrRow[r]; p < A_bsrRow[r + 1]; ++p) {
            int c = A_bsrCol[p];
            bool keep = lower ? (c <= r) : (c >= r);
            if (keep) T_bsrCol.push_back(c);
        }
        T_bsrRow[r + 1] = (int)T_bsrCol.size();
    }
}

static void build_map_bcsc2bcsr_host(
    int nb,
    const std::vector<int>& bcscColPtr,
    const std::vector<int>& bcscRowInd,
    const std::vector<int>& bcsrRowPtr,
    const std::vector<int>& bcsrColInd,
    std::vector<int>& map_bcsc2bcsr)
{
    int nnzb = (int)bcscRowInd.size();
    map_bcsc2bcsr.assign(nnzb, -1);

    for (int c = 0; c < nb; ++c) {
        for (int k = bcscColPtr[c]; k < bcscColPtr[c + 1]; ++k) {
            int r = bcscRowInd[k];
            int found = -1;
            for (int p = bcsrRowPtr[r]; p < bcsrRowPtr[r + 1]; ++p) {
                if (bcsrColInd[p] == c) {
                    found = p;
                    break;
                }
            }
            if (found < 0) {
                std::cerr << "Pattern mismatch at (" << r << "," << c << ")\n";
                std::exit(1);
            }
            map_bcsc2bcsr[k] = found;
        }
    }
}

template <typename T>
__global__ void reorder_bcsc_val_to_bcsr_bs5(
    int nnzb,
    const int* __restrict__ map_bcsc2bcsr,
    const T* __restrict__ bcsc_val,
    T* __restrict__ bcsr_val)
{
    int k = blockIdx.x;
    int t = threadIdx.x;
    if (k >= nnzb) return;

    int dst = map_bcsc2bcsr[k];
    const T* src = bcsc_val + k   * 25;
    T*       out = bcsr_val + dst * 25;

    for (int i = t; i < 25; i += blockDim.x) {
        out[i] = src[i];
    }
}
