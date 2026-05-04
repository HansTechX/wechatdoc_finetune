#!/usr/bin/env python3
"""
DSW 环境信息查询脚本
用法: python check_dsw_env.py > dsw_env_info.txt
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def run_command(cmd):
    """执行命令并返回输出"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception as e:
        return f"执行失败: {e}"


def get_system_info():
    """获取系统信息"""
    info = []
    info.append("=" * 50)
    info.append("           DSW 环境信息查询")
    info.append("=" * 50)
    info.append("")

    info.append("【系统信息】")
    info.append("-" * 40)
    info.append(f"操作系统: {platform.platform()}")
    info.append(f"Python 版本: {sys.version}")
    info.append(f"Python 路径: {sys.executable}")
    info.append("")

    return info


def get_cuda_info():
    """获取 CUDA 信息"""
    info = []
    info.append("【CUDA 信息】")
    info.append("-" * 40)

    # nvcc 版本
    nvcc_version = run_command("nvcc --version 2>&1 | grep release")
    if nvcc_version:
        info.append(f"nvcc: {nvcc_version}")

    # CUDA 环境变量
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        info.append(f"CUDA_HOME: {cuda_home}")

    if not nvcc_version and not cuda_home:
        info.append("未找到 CUDA 信息")

    info.append("")
    return info


def get_gpu_info():
    """获取 GPU 信息"""
    info = []
    info.append("【GPU 信息】")
    info.append("-" * 40)

    try:
        import torch
        info.append(f"PyTorch CUDA 可用: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            info.append(f"CUDA 版本: {torch.version.cuda}")
            info.append(f"GPU 数量: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                info.append(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                info.append(f"    显存: {props.total_memory / 1024**3:.1f} GB")
        else:
            info.append("CUDA 不可用")
    except ImportError:
        info.append("PyTorch 未安装，无法获取 GPU 信息")

    info.append("")
    return info


def get_package_info():
    """获取关键依赖版本"""
    info = []
    info.append("【关键依赖版本】")
    info.append("-" * 40)

    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("accelerate", "accelerate"),
        ("bitsandbytes", "bitsandbytes"),
        ("llamafactory", "llamafactory"),
        ("openpyxl", "openpyxl"),
        ("yaml", "pyyaml"),
        ("tensorboard", "tensorboard"),
        ("vllm", "vllm"),
        ("sglang", "sglang"),
    ]

    for module_name, package_name in packages:
        try:
            if module_name == "yaml":
                mod = __import__("yaml")
            else:
                mod = __import__(module_name)
            version = getattr(mod, "__version__", "未知版本")
            info.append(f"{package_name}: {version}")
        except ImportError:
            info.append(f"{package_name}: 未安装")

    info.append("")
    return info


def get_pip_list():
    """获取已安装的包列表"""
    info = []
    info.append("【已安装的 Python 包（全部）】")
    info.append("-" * 40)

    try:
        result = subprocess.run(
            ["pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            for line in lines:
                info.append(line)
        else:
            info.append("获取失败")
    except Exception as e:
        info.append(f"获取失败: {e}")

    info.append("")
    return info


def get_dsw_image_info():
    """获取 DSW 镜像信息"""
    info = []
    info.append("【DSW 镜像信息】")
    info.append("-" * 40)

    # 检查常见的 DSW 环境配置文件
    dsw_paths = [
        "/opt/dsw-env.yaml",
        "/root/.dsw-env",
        "/etc/dsw-image-info",
        "/root/.dsw_image_info",
    ]

    found = False
    for path in dsw_paths:
        if os.path.exists(path):
            info.append(f"找到配置文件: {path}")
            try:
                with open(path, "r") as f:
                    content = f.read()
                    if content.strip():
                        info.append(content)
                found = True
            except Exception as e:
                info.append(f"读取失败: {e}")

    if not found:
        info.append("未找到 DSW 环境配置文件")

    # 尝试获取镜像信息
    image_info = run_command("cat /etc/image-version 2>/dev/null")
    if image_info:
        info.append(f"镜像版本: {image_info}")

    info.append("")
    return info


def get_env_variables():
    """获取相关环境变量"""
    info = []
    info.append("【相关环境变量】")
    info.append("-" * 40)

    env_keys = [
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDA_VERSION",
        "LD_LIBRARY_PATH",
        "PYTHONPATH",
        "PATH",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
    ]

    for key in env_keys:
        value = os.environ.get(key)
        if value:
            # 截断过长的路径
            if len(value) > 100:
                value = value[:50] + "..." + value[-47:]
            info.append(f"{key}={value}")

    info.append("")
    return info


def main():
    """主函数"""
    output = []

    output.extend(get_system_info())
    output.extend(get_cuda_info())
    output.extend(get_gpu_info())
    output.extend(get_package_info())
    output.extend(get_dsw_image_info())
    output.extend(get_env_variables())
    output.extend(get_pip_list())

    # 打印到控制台
    print("\n".join(output))


if __name__ == "__main__":
    main()
