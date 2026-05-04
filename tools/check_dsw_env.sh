#!/bin/bash
# DSW 环境信息查询脚本

echo "=================================================="
echo "           DSW 环境信息查询"
echo "=================================================="
echo ""

# 1. 系统信息
echo "【系统信息】"
echo "----------------------------------------"
echo "操作系统: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "内核版本: $(uname -r)"
echo "架构: $(uname -m)"
echo ""

# 2. Python 信息
echo "【Python 信息】"
echo "----------------------------------------"
python --version
echo "Python 路径: $(which python)"
echo ""

# 3. CUDA 信息
echo "【CUDA 信息】"
echo "----------------------------------------"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    echo "nvcc 未找到"
fi
echo ""

# 4. GPU 信息
echo "【GPU 信息】"
echo "----------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi 未找到"
fi
echo ""

# 5. PyTorch 信息
echo "【PyTorch 信息】"
echo "----------------------------------------"
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'cuDNN 版本: {torch.backends.cudnn.version()}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"
echo ""

# 6. 关键依赖版本
echo "【关键依赖版本】"
echo "----------------------------------------"
python -c "
packages = [
    'llamafactory',
    'transformers',
    'peft',
    'accelerate',
    'bitsandbytes',
    'openpyxl',
    'pyyaml',
    'tensorboard',
    'vllm',
    'sglang',
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        version = getattr(mod, '__version__', '未知版本')
        print(f'{pkg}: {version}')
    except ImportError:
        print(f'{pkg}: 未安装')
"
echo ""

# 7. 镜像信息（DSW 特有）
echo "【镜像信息】"
echo "----------------------------------------"
if [ -f /opt/dsw-env.yaml ]; then
    echo "DSW 环境配置: /opt/dsw-env.yaml"
    cat /opt/dsw-env.yaml 2>/dev/null || echo "无法读取"
elif [ -f /root/.dsw-env ]; then
    echo "DSW 环境配置: /root/.dsw-env"
    cat /root/.dsw-env 2>/dev/null || echo "无法读取"
else
    echo "未找到 DSW 环境配置文件"
fi
echo ""

# 8. pip list 精简版
echo "【已安装的 Python 包（前 50 个）】"
echo "----------------------------------------"
pip list | head -50
echo ""

# 9. 环境变量
echo "【相关环境变量】"
echo "----------------------------------------"
env | grep -E "(CUDA|PYTHON|PATH|LD_LIBRARY)" | sort
echo ""
