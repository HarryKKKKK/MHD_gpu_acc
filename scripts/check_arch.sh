#!/bin/bash -l
#SBATCH -J check_gpu
#SBATCH -A MPHIL-NIKIFORAKIS-HK597-SL2-GPU
#SBATCH -p ampere
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 06:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

echo "========================================"
echo "CUDA GPU Architecture Checker"
echo "========================================"

# ------------------------------------------------------------
# 1. 检查 nvidia-smi
# ------------------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found."
    echo "Please run this script on a GPU compute node."
    exit 1
fi

echo
echo "[1] GPU information"
nvidia-smi --query-gpu=index,name,driver_version \
    --format=csv,noheader

# ------------------------------------------------------------
# 2. 获取 Compute Capability
# 优先使用 nvidia-smi；失败时尝试 PyTorch
# ------------------------------------------------------------
COMPUTE_CAP=""

if nvidia-smi --query-gpu=compute_cap \
    --format=csv,noheader 2>/dev/null | head -n 1 | grep -Eq '^[0-9]+\.[0-9]+'; then

    COMPUTE_CAP=$(
        nvidia-smi --query-gpu=compute_cap \
            --format=csv,noheader |
            head -n 1 |
            tr -d '[:space:]'
    )

    DETECTION_METHOD="nvidia-smi"

elif command -v python >/dev/null 2>&1 &&
     python -c "import torch; assert torch.cuda.is_available()" \
        >/dev/null 2>&1; then

    COMPUTE_CAP=$(
        python - <<'PY'
import torch

major, minor = torch.cuda.get_device_capability(0)
print(f"{major}.{minor}")
PY
    )

    DETECTION_METHOD="PyTorch"

else
    # --------------------------------------------------------
    # 3. 根据常见 GPU 名称进行后备判断
    # --------------------------------------------------------
    GPU_NAME=$(
        nvidia-smi --query-gpu=name \
            --format=csv,noheader |
            head -n 1
    )

    case "$GPU_NAME" in
        *H100*|*H800*)
            COMPUTE_CAP="9.0"
            ;;
        *A100*|*A800*)
            COMPUTE_CAP="8.0"
            ;;
        *L40S*|*L40*)
            COMPUTE_CAP="8.9"
            ;;
        *RTX\ 4090*|*RTX\ 4080*)
            COMPUTE_CAP="8.9"
            ;;
        *RTX\ 3090*|*RTX\ 3080*|*A6000*)
            COMPUTE_CAP="8.6"
            ;;
        *)
            echo
            echo "ERROR: Could not determine Compute Capability."
            echo "GPU name: $GPU_NAME"
            echo
            echo "Try running:"
            echo 'python -c "import torch; print(torch.cuda.get_device_capability(0))"'
            exit 1
            ;;
    esac

    DETECTION_METHOD="GPU name fallback"
fi

# 9.0 -> 90；8.0 -> 80
SM_ARCH=$(echo "$COMPUTE_CAP" | tr -d '.')

echo
echo "[2] Architecture detection"
echo "Detection method  : $DETECTION_METHOD"
echo "Compute Capability: $COMPUTE_CAP"
echo "CUDA architecture : sm_$SM_ARCH"

# ------------------------------------------------------------
# 4. 检查 nvcc
# ------------------------------------------------------------
echo
echo "[3] CUDA compiler check"

if command -v nvcc >/dev/null 2>&1; then
    NVCC_PATH=$(command -v nvcc)
    echo "nvcc path: $NVCC_PATH"
    nvcc --version | tail -n 1

    if nvcc --list-gpu-arch >/dev/null 2>&1; then
        if nvcc --list-gpu-arch | grep -qx "compute_$SM_ARCH"; then
            echo "nvcc supports: sm_$SM_ARCH"
        else
            echo
            echo "WARNING: This nvcc does not appear to support sm_$SM_ARCH."
            echo "You may need to load a newer CUDA module."
            echo
            echo "Available architectures:"
            nvcc --list-gpu-arch
        fi
    else
        echo "Could not automatically query nvcc architecture support."
    fi
else
    echo "WARNING: nvcc is not available in the current environment."
    echo "You may need to load a CUDA module, for example:"
    echo "  module avail cuda"
    echo "  module load cuda"
fi

# ------------------------------------------------------------
# 5. 输出 Makefile 配置
# ------------------------------------------------------------
echo
echo "========================================"
echo "Recommended Makefile configuration"
echo "========================================"
echo
echo "CUDA_ARCH := -arch=sm_$SM_ARCH"
echo
echo "For example:"
echo "NVCCFLAGS := -O3 \$(CUDA_ARCH)"
echo