# HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2504.05897-b31b1b.svg)](https://arxiv.org/abs/2504.05897)&nbsp;
</div>

## Introduction

HybriMoE is a hybrid CPU-GPU scheduling and cache management system to improve the efficiency of MoE inference. It addresses the high latency overhead associated with on-demand expert loading and unbalanced hardware utilization through three key innovations:

- **Hybrid MoE CPU-GPU Scheduling**: An efficient hybrid scheduling algorithm for MoE inference that dynamically balances workloads across GPUs and CPUs.
- **Impact-driven prefetching**: A prefetching mechanism that simulates the potential impact of preloading experts from subsequent layers and prioritizes those with the higher expected gains.
- **MoE-specialized Cache Management**: An expert score-based caching strategy that prioritizes high-demand experts across layers to minimize cache misses.

## Installation

- CUDA 12.1 and above, if you didn't have it yet, you may install from [here](https://developer.nvidia.com/cuda-downloads).
  
  ```sh
  # Adding CUDA to PATH
  export PATH=/usr/local/cuda/bin:$PATH
  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
  export CUDA_PATH=/usr/local/cuda
  ```

    ```sh
  conda create --name hybrimoe python=3.11
  conda activate hybrimoe # you may need to run ‘conda init’ and reopen shell first
  ```

- init source code 
    
    ```sh
    git clone https://github.com/PKU-SEC-Lab/HybriMoE
    cd HybriMoE
    git submodule init
    git submodule update
    ```

- Compile and install (for Linux)
    
    ```
    bash install.sh
    ```

### Downloading Model Weights
```shell
mkdir DeepSeek-V2-Lite-Chat-GGUF
cd DeepSeek-V2-Lite-Chat-GGUF

wget https://huggingface.co/mzwing/DeepSeek-V2-Lite-Chat-GGUF/resolve/main/DeepSeek-V2-Lite-Chat.Q4_K_M.gguf -O DeepSeek-V2-Lite-Chat.Q4_K_M.gguf
```

### Running Example
```shell
python ktransformers/local_chat.py --model_path deepseek-ai/DeepSeek-V2-Lite-Chat --gguf_path ./DeepSeek-V2-Lite-Chat-GGUF --cache_size 16 --prefetch_size 0 --optimize_rule_path ktransformers/optimize/optimize_rules/DeepSeek-V2-Chat.yaml
```

## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```
@article{zhong2025hybrimoe,
  title={HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference},
  author={Zhong, Shuzhang and Sun, Yanfan and Liang, Ling and Wang, Runsheng and Huang, Ru and Li, Meng},
  journal={arXiv preprint arXiv:2504.05897},
  year={2025}
}

## Contact
If you have any questions, please raise a GitHub issue or contact us via email zsz@stu.pku.edu.cn.

