#!/usr/bin/env python3
"""
一键执行完整的微调流程：数据处理 → 训练 → 测试
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


# ─── 工具函数 ────────────────────────────────────────────────

def run_command(cmd: list, description: str) -> bool:
    """
    执行命令并返回是否成功

    Args:
        cmd: 命令列表（如 ["python", "script.py", "--arg", "value"]）
        description: 步骤描述

    Returns:
        bool: 是否成功
    """
    print(f"\n{'='*60}")
    print(f"【{description}】")
    print(f"{'='*60}")
    print(f"执行命令: {' '.join(cmd)}")
    print(f"{'-'*60}")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n[!] {description} 失败 (退出码: {result.returncode})")
        return False

    print(f"\n[v] {description} 完成")
    return True


def get_script_dir() -> str:
    """获取脚本所在目录"""
    return os.path.dirname(os.path.abspath(__file__))


# ─── 流程步骤 ────────────────────────────────────────────────

def step_prepare_data(input_file: str = None, prompt_id: str = None, output_dir: str = "data", raw_output: bool = False) -> bool:
    """
    步骤1: 数据准备

    Args:
        input_file: Excel 输入文件路径
        prompt_id: 提示词ID
        output_dir: 数据输出目录
        raw_output: 是否使用原始标签（不使用编码映射）
    """
    cmd = ["python", "step1_prepare.py"]
    if input_file:
        cmd.extend(["--input", input_file])
    if output_dir != "data":
        cmd.extend(["--output_dir", output_dir])
    if prompt_id:
        cmd.extend(["--prompt_id", prompt_id])
    if raw_output:
        cmd.append("--raw_output")

    return run_command(cmd, "步骤1: 数据准备")


def step_train(config: str = None) -> bool:
    """
    步骤2: 模型训练
    """
    cmd = ["python", "step2_train.py"]
    if config:
        cmd.extend(["--config", config])

    return run_command(cmd, "步骤2: 模型训练")


def step_test(model_path: str = None, max_samples: int = None) -> bool:
    """
    步骤3: 本地推理测试
    """
    cmd = ["python", "step3_test.py"]
    if model_path:
        cmd.extend(["--model_path", model_path])
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])

    return run_command(cmd, "步骤3: 本地推理测试")


def step_http_test(max_samples: int = None, skip_serve: bool = False, workers: int = 1) -> bool:
    """
    步骤4: HTTP API 测试
    """
    cmd = ["python", "step4_test_http.py"]
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    if skip_serve:
        cmd.append("--skip_serve")
    if workers and workers > 1:
        cmd.extend(["--workers", str(workers)])

    return run_command(cmd, "步骤4: HTTP API 测试")


# ─── 主流程 ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="一键执行完整的微调流程：数据处理 → 训练 → 测试",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程（使用默认配置）
  python run_pipeline.py

  # 指定提示词ID
  python run_pipeline.py --prompt_id 1

  # 跳过数据准备
  python run_pipeline.py --skip_prepare

  # 跳过 HTTP API 测试
  python run_pipeline.py --skip_http

  # HTTP 测试时跳过部署（服务已运行）
  python run_pipeline.py --skip_serve

  # 仅执行 HTTP API 测试
  python run_pipeline.py --only_http

  # 测试时限制用例数量
  python run_pipeline.py --max_test_samples 100
        """
    )

    # 流程控制参数
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Excel 数据文件路径（传递给 step1_prepare.py，默认选最新）"
    )
    parser.add_argument(
        "--prompt_id",
        type=str,
        default=None,
        help="提示词ID（传递给 step1_prepare.py）"
    )
    parser.add_argument(
        "--raw_output",
        action="store_true",
        help="使用原始标签（不使用编码映射，传递给 step1_prepare.py）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="训练配置文件路径（默认: config/train_config.yaml）"
    )

    # 步骤跳过参数
    parser.add_argument(
        "--skip_prepare",
        action="store_true",
        help="跳过数据准备步骤"
    )
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="跳过训练步骤"
    )
    parser.add_argument(
        "--skip_test",
        action="store_true",
        help="跳过本地推理测试步骤（step3）"
    )
    parser.add_argument(
        "--skip_http",
        action="store_true",
        help="跳过 HTTP API 测试步骤（step4）"
    )
    parser.add_argument(
        "--skip_serve",
        action="store_true",
        help="跳过模型服务部署（假设服务已运行，传递给 step4）"
    )

    # 仅执行某个步骤
    parser.add_argument(
        "--only_prepare",
        action="store_true",
        help="仅执行数据准备"
    )
    parser.add_argument(
        "--only_train",
        action="store_true",
        help="仅执行训练"
    )
    parser.add_argument(
        "--only_test",
        action="store_true",
        help="仅执行本地推理测试"
    )
    parser.add_argument(
        "--only_http",
        action="store_true",
        help="仅执行 HTTP API 测试"
    )

    # 测试参数
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="测试时限制用例数量"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="HTTP 测试并发线程数（默认 1）"
    )

    # 其他参数
    parser.add_argument(
        "--data_output_dir",
        type=str,
        default="data",
        help="数据输出目录（默认: data）"
    )

    args = parser.parse_args()

    # 处理互斥参数
    only_count = sum([args.only_prepare, args.only_train, args.only_test, args.only_http])
    if only_count > 1:
        print("[!] 错误: --only_prepare, --only_train, --only_test, --only_http 只能指定一个")
        sys.exit(1)

    # 设置工作目录
    script_dir = get_script_dir()
    os.chdir(script_dir)

    # 打印开始信息
    print(f"\n{'='*60}")
    print(f"【微调流程一键执行】")
    print(f"{'='*60}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"工作目录: {script_dir}")

    if args.prompt_id:
        print(f"提示词ID: {args.prompt_id}")
    if args.input:
        print(f"数据文件: {args.input}")
    if args.raw_output:
        print(f"输出模式: 原始标签（不使用编码映射）")

    # 执行流程
    success = True
    start_time = datetime.now()

    try:
        # 仅执行某个步骤
        if args.only_prepare:
            success = step_prepare_data(args.input, args.prompt_id, args.data_output_dir, args.raw_output)
        elif args.only_train:
            success = step_train(args.config)
        elif args.only_test:
            success = step_test(max_samples=args.max_test_samples)
        elif args.only_http:
            success = step_http_test(max_samples=args.max_test_samples, skip_serve=args.skip_serve, workers=args.workers)

        # 完整流程（可选择性跳过）
        else:
            # 步骤1: 数据准备
            if not args.skip_prepare:
                if not step_prepare_data(args.input, args.prompt_id, args.data_output_dir, args.raw_output):
                    success = False
            else:
                print("\n[跳过] 数据准备步骤")

            # 步骤2: 训练
            if success and not args.skip_train:
                if not step_train(args.config):
                    success = False
            elif args.skip_train:
                print("\n[跳过] 训练步骤")

            # 步骤3: 本地推理测试
            if success and not args.skip_test:
                if not step_test(max_samples=args.max_test_samples):
                    success = False
            elif args.skip_test:
                print("\n[跳过] 本地推理测试步骤")

            # 步骤4: HTTP API 测试
            if success and not args.skip_http:
                if not step_http_test(max_samples=args.max_test_samples, skip_serve=args.skip_serve, workers=args.workers):
                    success = False
            elif args.skip_http:
                print("\n[跳过] HTTP API 测试步骤")

    except KeyboardInterrupt:
        print(f"\n\n[!] 用户中断")
        success = False

    except Exception as e:
        print(f"\n\n[!] 异常: {e}")
        import traceback
        traceback.print_exc()
        success = False

    # 打印结束信息
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*60}")
    if success:
        print(f"[v] 流程执行完成")
    else:
        print(f"[x] 流程执行失败")
    print(f"{'='*60}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时:   {int(duration // 60)} 分 {int(duration % 60)} 秒")
    print(f"{'='*60}\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
