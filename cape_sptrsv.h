#include "common.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cusparse_v2.h>
#include <assert.h>

__global__
void spTrSolveL(const int *__restrict__ d_csrRowPtr,
                const int *__restrict__ d_csrColIdx,
                const VALUE_TYPE *__restrict__ d_csrVal,
                volatile bool *__restrict__ d_get_value,// 0*m
                const int m, // rows
                const VALUE_TYPE *__restrict__ d_b, // rhs
                VALUE_TYPE *d_x, // initVec
                int *d_id_extractor // 0
) {
//    const int global_id = atomicAdd(d_id_extractor, 1);
    const int global_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_id >= m) {
        return;
    }
    int col, j, i;
    VALUE_TYPE xi;
    VALUE_TYPE left_sum = static_cast<VALUE_TYPE>(0);
    i = global_id; // 3
    j = d_csrRowPtr[i];
    if (d_csrColIdx[j] > i) {
        return;
    }
    int end = d_csrRowPtr[i + 1] - 1;
    while (d_csrColIdx[end] > i) {
        end -= 1;
    }
    end += 1;
    while (j < end) { // 1,2
        col = d_csrColIdx[j];
        while (d_get_value[col] && j < end) {
            left_sum += d_csrVal[j] * d_x[col];
            j++;
            if (j < end) {
                col = d_csrColIdx[j];
            }
        }
        if (i == col || j == end) {
            xi = (d_b[i] - left_sum);
            d_x[i] = xi;
            __threadfence();
            d_get_value[i] = true;
            j++;
        }
    }
}

__global__
void spTrSolveU(const int *__restrict__ d_csrRowPtr,
                const int *__restrict__ d_csrColIdx,
                const VALUE_TYPE *__restrict__ d_csrVal,
                volatile bool *__restrict__ d_get_value,// 0*m
                const int m, // rows
                const VALUE_TYPE *__restrict__ d_b, // rhs
                VALUE_TYPE *d_x, // initVec
                int *d_id_extractor // 0
) {
//    int global_idx = atomicAdd(d_id_extractor, 1);
    int global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int global_id = m - global_idx - 1;
    if (global_id < 0) {
        return;
    }
    int col, j, i;
    col = -1;
    VALUE_TYPE xi;
    VALUE_TYPE right_sum = static_cast<VALUE_TYPE>(0);
    i = global_id; // 3
    j = d_csrRowPtr[i + 1] - 1;
    if (d_csrColIdx[j] < i) {
        return;
    }
    int end = d_csrRowPtr[i];
    while (d_csrColIdx[end] < i) {
        end += 1;
    }
    int itr = 0;
    while (j >= end && d_csrColIdx[j] >= i) { // 1,2
        itr += 1;
        col = d_csrColIdx[j];
        while (j >= end && d_csrColIdx[j] > i && d_get_value[col]) {
            right_sum += d_csrVal[j] * d_x[col];
            j--;
            if (j >= end && d_csrColIdx[j] > i) {
                col = d_csrColIdx[j];
            }
        }
        if (i == col || j == end) {
            xi = (d_b[i] - right_sum) / (d_csrVal[j]);
            d_x[i] = xi;
            __threadfence();
            d_get_value[i] = true;
            j--;
        }
    }
}

void csr_cape_SpTRSV(
    int m,                          // 矩阵的行数/列数
    int* csrRowPtrA,                // CSR格式的行指针数组
    int* csrColIdxA,                // CSR格式的列索引数组
    VALUE_TYPE* csrValA,            // CSR格式的非零值数组
    int nnz,                        // 非零元素数量
    VALUE_TYPE* b                   // 右侧向量，同时也是输出结果
) {
    struct timeval t1, t2;
    VALUE_TYPE *d_csrValA = nullptr;
    int *d_csrRowPtrA = nullptr;
    int *d_csrColIdxA = nullptr;
    VALUE_TYPE *d_x;
    VALUE_TYPE *d_y;
    VALUE_TYPE *d_z;

    cudaMalloc((void**)&d_csrValA, nnz * sizeof(VALUE_TYPE));
    cudaMalloc((void**)&d_csrRowPtrA, (m + 1) * sizeof(int));
    cudaMalloc((void**)&d_csrColIdxA, nnz * sizeof(int));

    cudaMemcpy(d_csrValA, csrValA, nnz * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrRowPtrA, csrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrColIdxA, csrColIdxA, nnz * sizeof(int), cudaMemcpyHostToDevice);

    // Vector d_x
    cudaMalloc((void **)&d_x, m * sizeof(VALUE_TYPE));
    cudaMemcpy(d_x, b, m * sizeof(VALUE_TYPE), cudaMemcpyHostToDevice);

    // Vector d_y
    cudaMalloc((void **)&d_y, m  * sizeof(VALUE_TYPE));
    cudaMemset(d_y, 0, m * sizeof(VALUE_TYPE));

    // Vector d_z
    cudaMalloc((void **)&d_z, m  * sizeof(VALUE_TYPE));
    cudaMemset(d_z, 0, m * sizeof(VALUE_TYPE));

    bool *d_get_value_L;
    cudaMalloc((void **) &d_get_value_L, (m) * sizeof(bool));
    cudaMemset(d_get_value_L, false, sizeof(bool) * m);

    bool *d_get_value_U;
    cudaMalloc((void **) &d_get_value_U, (m) * sizeof(bool));
    cudaMemset(d_get_value_U, false, sizeof(bool) * m);

    // step 5: solve L*y = x
    int num_threads = 32*32;
    int num_blocks = ceil((VALUE_TYPE) m / (VALUE_TYPE) (num_threads));

    int *d_id_extractor;
    cudaMalloc((void **) &d_id_extractor, sizeof(int));
    cudaMemset(d_x, 0, sizeof(VALUE_TYPE) * m);
    cudaMemset(d_id_extractor, 0, sizeof(int));

    gettimeofday(&t1, NULL);
    spTrSolveL<<< num_blocks, num_threads >>>
                (d_csrRowPtrA, d_csrColIdxA, d_csrValA,
                 d_get_value_L, m, d_x, d_z, d_id_extractor);
    spTrSolveU<<< num_blocks, num_threads >>>
                (d_csrRowPtrA, d_csrColIdxA, d_csrValA,
                 d_get_value_U, m, d_z, d_y, d_id_extractor);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    VALUE_TYPE time_cape_SV = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0);
    printf("\ncape csr ILU SV time: %f\n", (double)time_cape_SV);

    cudaFree(d_csrValA);
    cudaFree(d_csrRowPtrA);
    cudaFree(d_csrColIdxA);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_get_value_L);
    cudaFree(d_get_value_U);
}
