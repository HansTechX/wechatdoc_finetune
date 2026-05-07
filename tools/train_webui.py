#!/usr/bin/env python3
"""
LLaMA Factory WebUI 启动脚本

官方文档：https://github.com/hiyouga/LLaMA-Factory
使用方法：llamafactory-cli webui

在浏览器中进行可视化微调，支持训练、推理、导出等全流程操作。
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime


def setup_logger(log_dir: str) -> logging.Logger:
    """设置日志系统，与 step2_train.py 保持一致"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"webui_{timestamp}.log")

    logger = logging.getLogger("webui")
    logger.setLevel(logging.DEBUG)

    # 清除已有的 handlers
    logger.handlers.clear()

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 Handler (INFO 级别)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 文件 Handler (DEBUG 级别)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"日志文件: {log_file}")

    return logger


def main():
    parser = argparse.ArgumentParser(
        description="LLaMA Factory WebUI - 可视化微调界面",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python train_webui.py              # 默认启动（端口 7860）
  python train_webui.py --port 8080  # 指定端口

注意:
  - WebUI 启动后会在浏览器中打开操作界面
  - 所有训练参数在 WebUI 界面中配置，无需预设
  - 按 Ctrl+C 停止服务

官方文档:
  https://github.com/hiyouga/LLaMA-Factory
  https://llamafactory.readthedocs.io/
        """
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="WebUI 服务端口（默认：7860）"
    )

    args = parser.parse_args()

    # 生成运行名称：webui_<timestamp>
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"webui_{timestamp}"

    # 输出目录：outputs/<run_name>/logs/
    output_dir = os.path.join("outputs", run_name)
    log_dir = os.path.join(output_dir, "logs")

    # 设置日志
    logger = setup_logger(log_dir)

    # 打印启动信息
    logger.info("=" * 60)
    logger.info("  LLaMA Factory WebUI 启动")
    logger.info("=" * 60)
    logger.info(f"  运行名称: {run_name}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info(f"  服务端口: {args.port}")
    logger.info(f"  本地访问: http://127.0.0.1:{args.port}")
    logger.info(f"  DSW 访问: http://0.0.0.0:{args.port}")
    logger.info("")
    logger.info("  在 WebUI 界面中配置：")
    logger.info("    - 模型选择与路径")
    logger.info("    - 数据集与训练参数")
    logger.info("    - 推理与导出选项")
    logger.info("")
    logger.info("  按 Ctrl+C 停止服务")
    logger.info("=" * 60)
    logger.info("")

    # 构建命令
    cmd = ["llamafactory-cli", "webui", "--port", str(args.port)]
    logger.debug(f"执行命令: {' '.join(cmd)}")
    logger.info("")

    try:
        # 启动 WebUI 并实时捕获输出
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # 实时输出并记录日志
        for line in process.stdout:
            line = line.rstrip()
            if line:
                logger.info(line)

        # 等待进程结束
        process.wait()

        if process.returncode != 0:
            logger.error(f"WebUI 异常退出，退出码: {process.returncode}")
            sys.exit(process.returncode)

    except KeyboardInterrupt:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  WebUI 已停止")
        logger.info("=" * 60)
    except subprocess.CalledProcessError as e:
        logger.error(f"启动失败，退出码: {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        logger.error("找不到 'llamafactory-cli' 命令")
        logger.error("请确认已安装 LLaMA Factory:")
        logger.error("  pip install llamafactory")
        sys.exit(1)


if __name__ == "__main__":
    main()
