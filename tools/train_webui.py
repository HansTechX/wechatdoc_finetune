#!/usr/bin/env python3
"""
LLaMA Factory WebUI 启动脚本
通过浏览器进行可视化交互式训练
适用于：调试参数、可视化训练过程、快速实验
"""

import os
import sys

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

import argparse
import subprocess
from pathlib import Path

from step2_train import load_config, ensure_model_available, setup_logger


def main():
    parser = argparse.ArgumentParser(
        description="LLaMA Factory WebUI 可视化训练"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="训练配置文件路径（相对于项目根目录，默认：config/train_config.yaml）",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="运行名称",
    )
    parser.add_argument(
        "--skip_model_check",
        action="store_true",
        help="跳过模型检查（模型已下载时加速启动）",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建 Gradio 公开链接（适合远程访问）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="WebUI 服务端口（默认 7860）",
    )
    args = parser.parse_args()

    # 确定配置文件路径（相对于项目根目录）
    if args.config:
        # 如果是相对路径，转换为从项目根目录开始的绝对路径
        if not os.path.isabs(args.config):
            config_path = os.path.join(project_root, args.config)
        else:
            config_path = args.config
    else:
        config_path = os.path.join(project_root, "config/train_config.yaml")

    # 切换到项目根目录（确保相对路径正确）
    os.chdir(project_root)

    # 加载配置
    cfg = load_config(config_path)

    # 设置日志
    run_name = args.run_name or "webui"
    log_dir = os.path.join("outputs", run_name, "logs")
    logger = setup_logger(log_dir)

    logger.info("=" * 60)
    logger.info("  LLaMA Factory WebUI 启动")
    logger.info("=" * 60)
    logger.info(f"  项目根目录: {project_root}")
    logger.info(f"  配置文件: {config_path}")
    logger.info(f"  服务端口: {args.port}")
    logger.info(f"  跳过模型检查: {args.skip_model_check}")
    logger.info(f"  公开链接: {args.share}")

    # 模型检查（可选）
    if not args.skip_model_check:
        logger.info("")
        logger.info("  正在检查模型...")
        try:
            model_path = ensure_model_available(cfg, logger)
            logger.info(f"  ✓ 模型就绪: {model_path}")
        except Exception as e:
            logger.warning(f"  模型检查失败: {e}")
            logger.info("  将继续启动 WebUI（可能需要手动配置）")
    else:
        logger.info("")
        logger.info("  [跳过] 模型检查")

    # 启动 WebUI
    logger.info("")
    logger.info("=" * 60)
    logger.info("  启动 WebUI 服务...")
    logger.info("=" * 60)

    cmd = ["llamafactory-cli", "webui", "--port", str(args.port)]

    if args.share:
        cmd.append("--share")

    logger.info(f"  启动命令: {' '.join(cmd)}")
    logger.info("")
    logger.info("  访问地址: http://localhost:{args.port}")
    if args.share:
        logger.info("  公开链接将生成，可分享给他人访问")
    logger.info("")
    logger.info("  按 Ctrl+C 停止服务")
    logger.info("=" * 60)
    logger.info("")

    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  WebUI 已停止")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
