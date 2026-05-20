#include "common.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <cusparse_v2.h>
#include <assert.h>
constexpr int M = 8, N = 8, K = 4;
__device__ __forceinline__ void mma_m8n8k4(double *acc, double &frag_a, double &frag_b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1]):
        "d"(frag_a), "d"(frag_b)
    );
}
__global__ void bsr_spmv_kernel(const int *__restrict__ d_bsrRowPtr_A,
                                const int *__restrict__ d_bsrColIdx_A,
                                const VALUE_TYPE *__restrict__ d_bsrVal_A,
                                int mb,int nnzb,int bdim,
                                const VALUE_TYPE *__restrict__ d_Val_B,
                                VALUE_TYPE *d_Val_C){
        int warp_id = threadIdx.x / 32;          // 当前warp在block中的ID (0~3)
        int lane_id = threadIdx.x % 32;          // 当前thread在warp中的ID (0~31)
        int rows_per_block = blockDim.x / 32;  // 每个block处理的行数 (8)
        int start_row = blockIdx.x * rows_per_block + warp_id;
        if(start_row>=mb) return;
        int row=lane_id/4;
        int col=lane_id%4;
        int A_row_begin=d_bsrRowPtr_A[start_row];
        int A_row_end=d_bsrRowPtr_A[start_row+1];
        int A_colindex;
        int offset,gl_offset,B_offset;
        double fragA,fragB,fragC[2];
        fragC[0]=0.0,fragC[1]=0.0;
        for(int i=A_row_begin;i<A_row_end;i++){
            int col_now=col;
            A_colindex=d_bsrColIdx_A[i];
            //while(col_now<bdim){
            for(int j=0;j<((bdim+3)/4);j++){
                offset=row*bdim+col_now;
                gl_offset=i*bdim*bdim+offset;
                B_offset=A_colindex*bdim+col_now;
                //fragA = (row < bdim && col_now < bdim) ? d_bsrVal_A[gl_offset] : 0.0;
                //fragB = (row < bdim && col_now < bdim) ? d_Val_B[B_offset] : 0.0;
                fragA=d_bsrVal_A[gl_offset];
                //fragA = load_float_from_global(d_bsrVal_A + gl_offset);
                fragB = d_Val_B[B_offset];
                mma_m8n8k4(fragC, fragA, fragB);
                col_now+=4;
            }
        }
        //printf("blockrow=%d,lane=%d,%f,%f,\n",start_row,lane_id,fragC[0],fragC[1]);
        if(lane_id%4==0){
            d_Val_C[(start_row*bdim)+row]=fragC[0];
            //printf("blockrow=%d,lane=%d,startrow=%d,row=%d,%f,%f,\n",start_row,lane_id,start_row,row,fragC[0],fragC[1]);
        }
       // nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a_frag;
        //nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> b_frag;
       // nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> c_frag;
        

}
__global__ void bsr_spmv_kernel_new(const int *__restrict__ d_bsrRowPtr_A,
                                const int *__restrict__ d_bsrColIdx_A,
                                const VALUE_TYPE *__restrict__ d_bsrVal_A,
                                int mb,int nnzb,int bdim,
                                const VALUE_TYPE *__restrict__ d_Val_B,
                                VALUE_TYPE *d_Val_C){
        int warp_id = threadIdx.x / 32;          // 当前warp在block中的ID (0~3)
        int lane_id = threadIdx.x % 32;          // 当前thread在warp中的ID (0~31)
        int rows_per_block = blockDim.x / 32;  // 每个block处理的行数 (4)
        int start_row = blockIdx.x * rows_per_block + warp_id;
        if(start_row>=mb) return;
        int row=lane_id/4;
        int col=lane_id%4;
        int A_row_begin=d_bsrRowPtr_A[start_row];
        int A_row_end=d_bsrRowPtr_A[start_row+1];
        int A_colindex;
        int offset,gl_offset,B_offset;
        double fragA,fragB,fragC[2];
        fragC[0]=0.0,fragC[1]=0.0;
        for(int i=A_row_begin;i<A_row_end;i++){
            int col_now=col;
            A_colindex=d_bsrColIdx_A[i];
            //while(col_now<bdim){
            for(int j=0;j<((bdim+3)/4);j++){
                
                offset=row*bdim+col_now;
                gl_offset=i*bdim*bdim+offset;
                B_offset=A_colindex*bdim+col_now;
                fragA = (row < bdim && col_now < bdim) ? d_bsrVal_A[gl_offset] : 0.0;
                fragB = (row < bdim && col_now < bdim) ? d_Val_B[B_offset] : 0.0;
                //fragA=d_bsrVal_A[gl_offset];
                //fragA = load_float_from_global(d_bsrVal_A + gl_offset);
                //fragB = d_Val_B[B_offset];
                mma_m8n8k4(fragC, fragA, fragB);
                col_now+=4;
            }
        }
        //printf("blockrow=%d,lane=%d,%f,%f,\n",start_row,lane_id,fragC[0],fragC[1]);
        if(lane_id%4==0&&row<bdim){
            d_Val_C[(start_row*bdim)+row]=fragC[0];
            //printf("blockrow=%d,lane=%d,startrow=%d,row=%d,%f,%f,\n",start_row,lane_id,start_row,row,fragC[0],fragC[1]);
        }
       // nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::row_major> a_frag;
        //nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, M, N, K, nvcuda::wmma::precision::tf32, nvcuda::wmma::col_major> b_frag;
       // nvcuda::wmma::fragment<nvcuda::wmma::accumulator, M, N, K, float> c_frag;
        

}
__device__ __forceinline__ double load_double_cs(const double* p)
{
    double v;
    asm volatile("ld.global.cs.f64 %0, [%1];" : "=d"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ void store_double_cs(double* p, double v)
{
    asm volatile("st.global.cs.f64 [%0], %1;" :: "l"(p), "d"(v));
}

__device__ __forceinline__ int load_int_cs(const int* p)
{
    int v;
    asm volatile("ld.global.cs.s32 %0, [%1];" : "=r"(v) : "l"(p));
    return v;
}
__device__ __forceinline__ double load_ca(const double* p)
{
    double v;
    asm volatile("ld.global.ca.f64 %0, [%1];" : "=d"(v) : "l"(p));
    return v;
}
__global__ void bsr_spmv_bdim5_kernel(const int *__restrict__ d_bsrRowPtr_A,
                                      const int *__restrict__ d_bsrColIdx_A,
                                      const VALUE_TYPE *__restrict__ d_bsrVal_A,
                                      int mb,
                                      int nnzb,
                                      const VALUE_TYPE *__restrict__ d_Val_B,
                                      VALUE_TYPE *__restrict__ d_Val_C)
{
    // 一个warp处理一个BSR block row
    const int warp_id        = threadIdx.x >> 5;     // warp id in block
    const int lane_id        = threadIdx.x & 31;     // lane id in warp
    const int warps_per_blk  = blockDim.x >> 5;
    const int block_row      = blockIdx.x * warps_per_blk + warp_id;
    constexpr int BDIM = 5;
    if (block_row >= mb) return;

    // 线程布局：
    // row = 0..7, col = 0..3
    // 一个row对应4个lane
    const int row = lane_id >> 2;   // lane_id / 4
    const int col = lane_id & 3;    // lane_id % 4

    const int A_row_begin = d_bsrRowPtr_A[block_row];
    const int A_row_end   = d_bsrRowPtr_A[block_row + 1];

    // 这里只保留你原来的frag形式
    VALUE_TYPE fragA = (VALUE_TYPE)0;
    VALUE_TYPE fragB = (VALUE_TYPE)0;
    VALUE_TYPE fragC[2];
    fragC[0] = (VALUE_TYPE)0;
    fragC[1] = (VALUE_TYPE)0;

    // bdim = 5:
    //   前4列 -> Tensor Core
    //   第5列   -> CUDA Core
    constexpr int bdim = 5;

    for (int bi = A_row_begin; bi < A_row_end; ++bi)
    {
        const int blk_col = d_bsrColIdx_A[bi];
        //const int blk_col = load_int_cs(d_bsrColIdx_A+bi);
        const int A_base  = bi * BDIM*BDIM;
        const int x_base  = blk_col * BDIM;

        // ----------------------------
        // Tensor Core part: columns 0~3
        // ----------------------------
        if (row < BDIM) {
            //fragA = d_bsrVal_A[A_base + row * BDIM + col];
            fragA = load_double_cs(d_bsrVal_A + A_base + row * BDIM + col);
        }

        // 只给 B 的第一列对应位置填 x
        if (lane_id < 4) {
            fragB = d_Val_B[x_base + lane_id];
            //fragB = load_ca(d_Val_B+x_base + lane_id);
        } 

        // 这里默认你已有这个封装
        mma_m8n8k4(fragC, fragA, fragB);

        // ============================================================
        // Part 2. 第5列: k = 4 走 CUDA Core
        // 只让 col==0 的线程参与，避免重复
        // ============================================================
        if (row < bdim && col == 0)
        {
            const int k_tail = 4;

            //const int a_offset_tail = bi * bdim * bdim + row * bdim + k_tail;
            const VALUE_TYPE a_tail = load_double_cs(d_bsrVal_A + A_base + row * BDIM + 4);
            const int b_offset_tail = blk_col * bdim + k_tail;
           // const VALUE_TYPE b_tail= load_double_cs(d_Val_B+blk_col * bdim + 4);

            //fragC[0] += d_bsrVal_A[a_offset_tail] * d_Val_B[b_offset_tail];
            fragC[0] += a_tail*d_Val_B[b_offset_tail];
        }
    }

    // 写回结果
    // 仍然只让每个row的 col==0 线程写
    if (row < bdim && col == 0)
    {
        d_Val_C[block_row * bdim + row] = fragC[0];
        //store_double_cs(d_Val_C+block_row * bdim + row,fragC[0]);
    }
}

__global__ void bsr_spmv_bdim5_kernel_dbuff_Aonly(
    const int *__restrict__ d_bsrRowPtr_A,
    const int *__restrict__ d_bsrColIdx_A,
    const VALUE_TYPE *__restrict__ d_bsrVal_A,
    int mb,
    int nnzb,
    const VALUE_TYPE *__restrict__ d_Val_B,
    VALUE_TYPE *__restrict__ d_Val_C)
{
    // 一个warp处理一个BSR block row
    const int warp_id       = threadIdx.x >> 5;
    const int lane_id       = threadIdx.x & 31;
    const int warps_per_blk = blockDim.x >> 5;
    const int block_row     = blockIdx.x * warps_per_blk + warp_id;

    constexpr int BDIM = 5;
    constexpr int bdim = 5;

    if (block_row >= mb) return;

    // row = 0..7, col = 0..3
    const int row = lane_id >> 2;
    const int col = lane_id & 3;

    const int A_row_begin = d_bsrRowPtr_A[block_row];
    const int A_row_end   = d_bsrRowPtr_A[block_row + 1];

    VALUE_TYPE fragA_cur, fragA_next;
    VALUE_TYPE aTail_cur, aTail_next;
    VALUE_TYPE fragB;
    VALUE_TYPE fragC[2];

    fragA_cur  = (VALUE_TYPE)0;
    fragA_next = (VALUE_TYPE)0;
    aTail_cur  = (VALUE_TYPE)0;
    aTail_next = (VALUE_TYPE)0;
    fragC[0]   = (VALUE_TYPE)0;
    fragC[1]   = (VALUE_TYPE)0;

    if (A_row_begin >= A_row_end)
    {
        if (row < bdim && col == 0)
        {
            d_Val_C[block_row * bdim + row] = (VALUE_TYPE)0;
        }
        return;
    }

    // ----------------------------------------
    // preload 第一个 block 的 A 到 cur
    // ----------------------------------------
    {
        const int bi0    = A_row_begin;
        const int A_base = bi0 * BDIM * BDIM;   // = bi0 * 25

        if (row < BDIM)
        {
            fragA_cur = d_bsrVal_A[A_base + row * BDIM + col];
            if (col == 0)
            {
                aTail_cur = d_bsrVal_A[A_base + row * BDIM + 4];
            }
        }
    }

    for (int bi = A_row_begin; bi < A_row_end; ++bi)
    {
        const int blk_col = d_bsrColIdx_A[bi];
        const int x_base  = blk_col * BDIM;

        // ----------------------------------------
        // 保持你原来的 B 加载逻辑
        // ----------------------------------------
        if (lane_id < 4)
        {
            fragB = d_Val_B[x_base + lane_id];
        }
        else
        {
            fragB = (VALUE_TYPE)0;
        }

        // ----------------------------------------
        // 当前 block 用 cur 参与计算
        // ----------------------------------------
        mma_m8n8k4(fragC, fragA_cur, fragB);

        if (row < bdim && col == 0)
        {
            fragC[0] += aTail_cur * d_Val_B[x_base + 4];
        }

        // ----------------------------------------
        // 预取下一 block 的 A
        // 这里只预取 A，不碰 B
        // ----------------------------------------
        if (bi + 1 < A_row_end)
        {
            const int bi_next    = bi + 1;
            const int A_base_nxt = bi_next * BDIM * BDIM;   // = (bi+1) * 25

            // 先清零，避免 row>=BDIM 的线程吃到旧值
            fragA_next = (VALUE_TYPE)0;
            aTail_next = (VALUE_TYPE)0;

            if (row < BDIM)
            {
                fragA_next = d_bsrVal_A[A_base_nxt + row * BDIM + col];
                if (col == 0)
                {
                    aTail_next = d_bsrVal_A[A_base_nxt + row * BDIM + 4];
                }
            }

            // rotate
            fragA_cur = fragA_next;
            aTail_cur = aTail_next;
        }
    }

    // ----------------------------------------
    // 保持你原来的写回方式
    // ----------------------------------------
    if (row < bdim && col == 0)
    {
        d_Val_C[block_row * bdim + row] = fragC[0];
    }
}

__global__ void bsr_spmv_cuda_bdim5_kernel(const int* __restrict__ d_bsrRowPtr_A,
                                           const int* __restrict__ d_bsrColIdx_A,
                                           const VALUE_TYPE* __restrict__ d_bsrVal_A,
                                           int mb,
                                           int nnzb,
                                           const VALUE_TYPE* __restrict__ d_Val_B,
                                           VALUE_TYPE* __restrict__ d_Val_C)
{
    constexpr int BDIM = 5;

    // 一个 warp 处理一个 BSR block row
    const int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    const int warp_id_global   = global_thread_id >> 5;   // 全局 warp id
    const int lane_id          = threadIdx.x & 31;        // warp 内 lane id
    const int block_row        = warp_id_global;

    if (block_row >= mb) return;

    // 只用前 5 个 lane 分别计算 block-row 内的 5 个输出元素
    if (lane_id >= BDIM) return;

    const int local_row = lane_id;   // 0~4

    const int row_begin = d_bsrRowPtr_A[block_row];
    const int row_end   = d_bsrRowPtr_A[block_row + 1];

    VALUE_TYPE sum = static_cast<VALUE_TYPE>(0);

    for (int bi = row_begin; bi < row_end; ++bi)
    {
        const int blk_col = d_bsrColIdx_A[bi];
        const int A_base  = bi * BDIM * BDIM;
        const int x_base  = blk_col * BDIM;

        // 当前 5x5 block 的第 local_row 行 与 x 对应 5 维子向量做点积
        sum += d_bsrVal_A[A_base + local_row * BDIM + 0] * d_Val_B[x_base + 0];
        sum += d_bsrVal_A[A_base + local_row * BDIM + 1] * d_Val_B[x_base + 1];
        sum += d_bsrVal_A[A_base + local_row * BDIM + 2] * d_Val_B[x_base + 2];
        sum += d_bsrVal_A[A_base + local_row * BDIM + 3] * d_Val_B[x_base + 3];
        sum += d_bsrVal_A[A_base + local_row * BDIM + 4] * d_Val_B[x_base + 4];
    }

    d_Val_C[block_row * BDIM + local_row] = sum;
}

void bsr_SpMV_tensorcore(const int           *bsrRowPtr_A,
                       const int           *bsrColIdx_A,
                       const VALUE_TYPE    *bsrVal_A,
                       const int mb,
                       const int nnzb,
                       const int bdim,
                       const VALUE_TYPE    *Val_B,
                        VALUE_TYPE          *Val_C,
                        VALUE_TYPE  *res_Y ){
    struct timeval t1, t2;
        int *d_bsrRowPtr_A;
    int *d_bsrColIdx_A;
    VALUE_TYPE *d_bsrVal_A;
    VALUE_TYPE *d_Val_B;
    VALUE_TYPE *d_Val_C;
    VALUE_TYPE *h_Val_C = (VALUE_TYPE *)malloc(sizeof(VALUE_TYPE) * mb * bdim);
    // 计算需要分配的内存大小
    size_t bsrRowPtr_size = (mb + 1) * sizeof(int);
    size_t bsrColIdx_size = nnzb * sizeof(int);
    size_t bsrVal_size = nnzb * bdim * bdim * sizeof(VALUE_TYPE);
    size_t Val_B_size = mb * bdim * sizeof(VALUE_TYPE);
    size_t Val_C_size = mb * bdim * sizeof(VALUE_TYPE);
    // 分配设备内存
    cudaMalloc((void**)&d_bsrRowPtr_A, bsrRowPtr_size);
    cudaMalloc((void**)&d_bsrColIdx_A, bsrColIdx_size);
    cudaMalloc((void**)&d_bsrVal_A, bsrVal_size);
    cudaMalloc((void**)&d_Val_B, Val_B_size);
    cudaMalloc((void**)&d_Val_C, Val_C_size);
    
    // 将数据从主机复制到设备
    cudaMemcpy(d_bsrRowPtr_A, bsrRowPtr_A, bsrRowPtr_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bsrColIdx_A, bsrColIdx_A, bsrColIdx_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bsrVal_A, bsrVal_A, bsrVal_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Val_B, Val_B, Val_B_size, cudaMemcpyHostToDevice);
    
    // 确保内存分配和拷贝成功
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        // 这里应该添加错误处理代码，比如释放已分配的内存
    }
    
    const int warps_per_block = 4;  // 每个block 4 warps (128 threads)
    const int rows_per_block = warps_per_block * 1;  // 每个block处理 4 行 (4 warps × 1 rows/warp)
    dim3 blockDim(32 * warps_per_block);  // 128 threads per block
    dim3 gridDim((mb + rows_per_block - 1) / rows_per_block);  // ceil(mb / 8)
    printf("mb%d\n",mb);
    //warm-up
    for (int i = 0; i < 100; ++i){
        //bsr_spmv_kernel_new<<<gridDim, blockDim>>>(d_bsrRowPtr_A,d_bsrColIdx_A,d_bsrVal_A,mb,nnzb,bdim,d_Val_B,d_Val_C);
        bsr_spmv_bdim5_kernel<<<gridDim, blockDim>>>(d_bsrRowPtr_A,d_bsrColIdx_A,d_bsrVal_A,mb,nnzb,d_Val_B,d_Val_C);
    }
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    for (int i = 0; i < 1000; ++i){
        //bsr_spmv_kernel_new<<<gridDim, blockDim>>>(d_bsrRowPtr_A,d_bsrColIdx_A,d_bsrVal_A,mb,nnzb,bdim,d_Val_B,d_Val_C);
        bsr_spmv_bdim5_kernel<<<gridDim, blockDim>>>(d_bsrRowPtr_A,d_bsrColIdx_A,d_bsrVal_A,mb,nnzb,d_Val_B,d_Val_C);
    }
    // 最后记得释放设备内存
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    double time_tensor_SV = ((t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0)/1000;
    printf("tensor-core SpMV time: %f\n",time_tensor_SV);
    long long int totalFlops = 2 * nnzb * bdim * bdim;  // Total number of floating point operations
    double gflops = (totalFlops / time_tensor_SV) / 1e6;  // GFLOPS
    printf("tensor-core GFLOPS: %f\n", gflops);
    cudaMemcpy(h_Val_C,d_Val_C,mb*bdim*sizeof(VALUE_TYPE),cudaMemcpyDeviceToHost);

    memcpy(res_Y,h_Val_C,mb*bdim*sizeof(VALUE_TYPE));

    free(h_Val_C);
    h_Val_C= NULL;
    cudaFree(d_bsrRowPtr_A);
    cudaFree(d_bsrColIdx_A);
    cudaFree(d_bsrVal_A);
    cudaFree(d_Val_B);
    cudaFree(d_Val_C);
}