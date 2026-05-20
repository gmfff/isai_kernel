cmake_minimum_required(VERSION 3.18)
project(gko_ilu_trisolve_gpu CXX)

#find_package(Ginkgo 
   #         REQUIRED
   #         CONFIG
   #         PATHS
   #         /home/guomingfeng/local/ginkgo/)

set(CMAKE_CUDA_ARCHITECTURES 80 CACHE STRING "")

enable_language(CUDA)

add_executable(tri_solve_tensor main.cu)

# 设置 VALUE_TYPE 为 double
target_compile_definitions(tri_solve_tensor PRIVATE USE_DOUBLE)


target_compile_options(tri_solve_tensor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        -w
        -m64
        -Xptxas=-v,-dlcm=cg
    >
)


# 链接 Ginkgo
#target_link_libraries(tri_solve_tensor PRIVATE Ginkgo::ginkgo)

# 手动加 cudart （CUDA runtime）
find_package(CUDAToolkit REQUIRED)
target_link_libraries(tri_solve_tensor PRIVATE CUDA::cusparse CUDA::cudart CUDA::cublas)

set_property(TARGET tri_solve_tensor PROPERTY CXX_STANDARD 17)
