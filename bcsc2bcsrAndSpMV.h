#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <cassert>
#include "common.h"

// ------------------ GPU kernels ------------------

// 1. 展开 block_col
__global__
void expand_block_col(int NB, const int* col_ptr, int* block_col)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= NB) return;
    for (int p = col_ptr[j]; p < col_ptr[j+1]; ++p)
        block_col[p] = j;
}

// 2. 统计每行非零块数
__global__
void count_rows(int nnzb, const int* row_ind, int* row_counts)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnzb) return;
    atomicAdd(&row_counts[row_ind[p]], 1);
}

// 3. scatter blocks 到 BCSR
__global__
void scatter_blocks(int nnzb, int bs, const int* row_ind, const int* block_col,
                    const VALUE_TYPE* in_val, int* row_next, int* col_ind, VALUE_TYPE* out_val)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= nnzb) return;

    int i = row_ind[p];
    int j = block_col[p];

    int dst = atomicAdd(&row_next[i], 1);
    col_ind[dst] = j;

    for (int k = 0; k < bs; ++k)
        out_val[dst*bs + k] = in_val[p*bs + k]; // 块内 row-major 不变
}

// 4. BCSR SpMV kernel
__global__
void bcsr_spmv(int MB, int R, int C,
               const int* row_ptr, const int* col_ind, const VALUE_TYPE* val,
               const VALUE_TYPE* x, VALUE_TYPE* y)
{
    int br = blockIdx.x * blockDim.x + threadIdx.x; // block row index
    if (br >= MB) return;

    int row_start = row_ptr[br];
    int row_end   = row_ptr[br+1];

    for (int r = 0; r < R; ++r)
    {
        VALUE_TYPE sum = 0.0;

        for (int idx = row_start; idx < row_end; ++idx)
        {
            int bc = col_ind[idx];
            const VALUE_TYPE* block = &val[idx * R * C];

            for (int c = 0; c < C; ++c)
                sum += block[r*C + c] * x[bc*C + c];
        }
        y[br*R + r] = sum;
    }
}

// ------------------ Host helper ------------------
void bcsc_to_bcsr(int MB, int NB, int R, int C,
                  const int* d_MbrowU, const int* d_McolU, const VALUE_TYPE* d_MvalU,
                  int* d_row_ptr, int* d_col_ind, VALUE_TYPE* d_val)
{
    int nnzb;
    cudaMemcpy(&nnzb, &d_MbrowU[NB], sizeof(int), cudaMemcpyDeviceToHost);

    int* d_block_col;
    cudaMalloc(&d_block_col, nnzb * sizeof(int));

    // 1. 展开 block_col
    int tpb = 128;
    int nblk = (NB + tpb -1)/tpb;
    expand_block_col<<<nblk, tpb>>>(NB, d_MbrowU, d_block_col);

    // 2. count rows
    int* d_row_counts;
    cudaMalloc(&d_row_counts, MB * sizeof(int));
    cudaMemset(d_row_counts, 0, MB * sizeof(int));
    nblk = (nnzb + tpb -1)/tpb;
    count_rows<<<nblk, tpb>>>(nnzb, d_McolU, d_row_counts);

    // 3. exclusive scan -> row_ptr
    void* d_temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_row_counts, d_row_ptr, MB);
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_row_counts, d_row_ptr, MB);
    cudaFree(d_temp);

    // 4. scatter blocks
    int* d_row_next;
    cudaMalloc(&d_row_next, MB * sizeof(int));
    cudaMemcpy(d_row_next, d_row_ptr, MB * sizeof(int), cudaMemcpyDeviceToDevice);

    scatter_blocks<<<nblk, tpb>>>(nnzb, R*C, d_McolU, d_block_col,
                                  d_MvalU, d_row_next, d_col_ind, d_val);

    cudaFree(d_block_col);
    cudaFree(d_row_counts);
    cudaFree(d_row_next);
}

void cpu_bcsr_spmv(int MB, int R, int C,
                   const int* row_ptr, const int* col_ind, const VALUE_TYPE* val,
                   const VALUE_TYPE* x, VALUE_TYPE* y)
{
    for(int br=0; br<MB; ++br)
    {
        for(int r=0;r<R;++r)
        {
            VALUE_TYPE sum = 0.0;
            for(int idx=row_ptr[br]; idx<row_ptr[br+1]; ++idx)
            {
                int bc = col_ind[idx];
                const VALUE_TYPE* block = &val[idx*R*C];
                for(int c=0;c<C;++c)
                    sum += block[r*C + c]*x[bc*C+c];
            }
            y[br*R+r] = sum;
        }
    }
}