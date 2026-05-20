// csr2bsr.h
#pragma once

#include "common.h"

#include <vector>
#include <map>
#include <array>
#include <iostream>
#include "config.h"

// CSR -> BSR with block size BS=5
// 要求 nrows == ncols == n，并且 n % BS == 0
inline void csr_to_bsr5(
    int n,                                  // nrows == ncols
    const std::vector<int>& csrRowPtr,      // size n+1
    const std::vector<int>& csrColInd,      // size nnz
    const std::vector<VALUE_TYPE>& csrVal,  // size nnz
    int& nb,                                // 输出: block 维度 = n / BS
    std::vector<int>& bsrRowPtr,            // 输出: size nb+1
    std::vector<int>& bsrColInd,            // 输出: size nnzb_blocks
    std::vector<VALUE_TYPE>& bsrVal         // 输出: size nnzb_blocks * BS2
)
{
    if (n % BS != 0) {
        std::cerr << "csr_to_bsr5: n = " << n
                  << " is not multiple of block size " << BS << std::endl;
        std::exit(EXIT_FAILURE);
    }

    nb = n / BS;
    bsrRowPtr.assign(nb + 1, 0);
    bsrColInd.clear();
    bsrVal.clear();

    // 对每个 block row 单独构造：用 map<blockCol, 5x5块>
    for (int br = 0; br < nb; ++br) {
        std::map<int, std::array<VALUE_TYPE, BS2>> blocks;

        // 这个 block row 对应的全局行区间 [br*BS, (br+1)*BS)
        for (int lr = 0; lr < BS; ++lr) {     // local row
            int row = br * BS + lr;          // global row
            int rowStart = csrRowPtr[row];
            int rowEnd   = csrRowPtr[row + 1];

            for (int p = rowStart; p < rowEnd; ++p) {
                int gc = csrColInd[p];       // global col
                VALUE_TYPE v = csrVal[p];

                int bc = gc / BS;           // block column
                int lc = gc % BS;           // local col in block

                auto it = blocks.find(bc);
                if (it == blocks.end()) {
                    std::array<VALUE_TYPE, BS2> blk{};
                    blk.fill(static_cast<VALUE_TYPE>(0.0));
                    blk[lr * BS + lc] = v;
                    blocks.emplace(bc, blk);
                } else {
                    it->second[lr * BS + lc] += v; // 累加
                }
            }
        }

        // 写入这一块行的所有块
        bsrRowPtr[br + 1] = bsrRowPtr[br] + (int)blocks.size();
        for (auto& kv : blocks) {
            int bc = kv.first;
            auto& blk = kv.second;
            bsrColInd.push_back(bc);
            for (int i = 0; i < BS2; ++i) {
                bsrVal.push_back(blk[i]);
            }
        }
    }
}
