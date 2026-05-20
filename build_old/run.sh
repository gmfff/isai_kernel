#!/bin/bash
#SBATCH --job-name=tri_test      # 任务名称
#SBATCH --partition=normal       # 分区名称（根据你之前的截图，应该是 normal）
#SBATCH --nodelist=cn2
#SBATCH --gres=gpu:1             # 申请 1 个 GPU
#SBATCH --output=tri_%j.out      # 标准输出日志（%j会被替换为作业ID）
#SBATCH --error=tri_%j.err       # 错误日志

# --- 关键步骤：加载 Conda 环境 ---

# 1. 初始化 Conda（防止找不到 conda 命令）
# 注意：这里假设你用的是 Miniconda/Anaconda，路径可能需要根据实际安装位置调整
# 如果不确定路径，通常 source ~/.bashrc 也可以
#source ~/.bashrc

# 2. 激活你的环境
source /home/guomingfeng/miniconda3/bin/activate
source ~/.bashrc
conda activate cuda12


echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Workdir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

nvidia-smi
nvidia-smi -L

# 可选：看是否是 MIG
nvidia-smi -q | grep -i mig -A 2
# --- 运行你的程序 ---

# 建议先打印一下当前路径，方便排查文件找不到的问题
echo "当前工作目录: $(pwd)"

# 运行命令
#srun ./spmv_tensor ../matrix_data/GT01R.mtx 5
#srun ./spmv_tensor ../matrix_data/small_block5_matrix.mtx 5
#srun ./spmv_tensor ../ginkgo_ISAI/case_20231024/matrix_2.mtx 5
./tri_solve_tensor ../../matrix_data/GT01R.mtx