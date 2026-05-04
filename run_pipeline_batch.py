#!/usr/bin/env python3
"""
批量执行多个微调任务
自动修改 train_config.yaml 并按顺序执行多个训练任务
训练完成后自动对比分析所有模型的效果
"""

import os
import sys
import time
import yaml
import json
import argparse
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional


# ============================================================
#  配置区域 - 修改此处定义默认任务
# ============================================================

# 默认任务列表：(模型名称, prompt_id)
# 任务名称规则：
#   - 普通模型：自动使用 LoRA
#   - 包含 "-4bit" 或 "bnb-4bit"：自动使用 QLoRA 4bit
#   - 包含 "-8bit" 或 "bnb-8bit"：自动使用 QLoRA 8bit
DEFAULT_TASKS: List[Tuple[str, str]] = [
    # 原始模型（非量化）
    # ("Qwen/Qwen3-4B", "001"),
    # ("Qwen/Qwen3-4B", "004"),
    # ("Qwen/Qwen3-8B", "001"),
    # ("Qwen/Qwen3-8B", "004"),
    # Unsloth 量化模型
    # ("unsloth/Qwen3-4B-unsloth-bnb-4bit", "001"),
    ("unsloth/Qwen3-4B-unsloth-bnb-4bit", "004"),
    # ("unsloth/Qwen3-8B-unsloth-bnb-4bit", "001"),
    # ("unsloth/Qwen3-8B-unsloth-bnb-4bit", "004"),
]

# 默认配置
DEFAULT_CONFIG_PATH = "config/train_config.yaml"
DEFAULT_INTERVAL_MINUTES = 5


# ============================================================
#  工具函数
# ============================================================

def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config_path: str, config: dict):
    """保存配置文件"""
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


def detect_quantization_config(model_name: str) -> Dict[str, Optional[int]]:
    """
    根据模型名称自动检测量化配置

    LLaMA Factory 量化规则：
    - finetuning_type 始终为 "lora"
    - 通过 quantization_bit (4/8) 控制量化

    Args:
        model_name: 模型名称或路径

    Returns:
        dict: {"quant_bits": 4/8/None}
    """
    model_lower = model_name.lower()

    # 检测 4bit 量化
    if "-4bit" in model_lower or "4bit" in model_lower or "bnb-4bit" in model_lower:
        return {"quant_bits": 4}

    # 检测 8bit 量化
    if "-8bit" in model_lower or "8bit" in model_lower or "bnb-8bit" in model_lower:
        return {"quant_bits": 8}

    # 非量化模型
    return {"quant_bits": None}


def update_config(config_path: str, model_name: str, prompt_id: Optional[str] = None) -> Dict[str, Optional[int]]:
    """
    更新配置文件中的模型名称、prompt_id 和量化配置

    自动根据模型名称判断是否为量化模型，并设置相应的 quantization 配置
    注意：finetuning_type 始终保持为 "lora"，量化通过 quantization_bit 控制

    Returns:
        量化配置信息 {"quant_bits": 4/8/None}
    """
    config = load_config(config_path)

    # 更新模型名称
    config["model"]["name_or_path"] = model_name

    # 更新 prompt_id（如果提供）
    if prompt_id is not None:
        config["data"]["prompt_id"] = prompt_id

    # finetuning_type 始终为 "lora"（LLaMA Factory 通过 quantization_bit 控制量化）
    config["training"]["finetuning_type"] = "lora"

    # 自动检测量化配置
    quant_config = detect_quantization_config(model_name)

    # 更新量化配置
    if quant_config["quant_bits"] is not None:
        config["quantization"]["enable"] = True
        config["quantization"]["bits"] = quant_config["quant_bits"]
    else:
        config["quantization"]["enable"] = False
        config["quantization"]["bits"] = 4  # 默认值（未启用时不影响）

    save_config(config_path, config)
    return quant_config


def parse_tasks_string(tasks_str: str) -> List[Tuple[str, str]]:
    """
    解析命令行传入的任务字符串

    Args:
        tasks_str: 格式 "模型1:prompt1,模型2:prompt2,..."

    Returns:
        [(model_name, prompt_id), ...]
    """
    tasks = []
    for task in tasks_str.split(","):
        parts = task.strip().split(":")
        if len(parts) == 2:
            tasks.append((parts[0].strip(), parts[1].strip()))
        else:
            print(f"[!] 跳过无效任务格式: {task}")
    return tasks


def run_pipeline(prompt_id: Optional[str] = None) -> bool:
    """执行单个训练任务"""
    cmd = ["python", "run_pipeline.py"]
    if prompt_id:
        cmd.extend(["--prompt_id", prompt_id])

    print(f"\n执行命令: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def find_latest_result_file(model_name: str, prompt_id: str) -> Optional[Dict]:
    """
    查找最新的 HTTP API 推理结果文件并提取指标（固定使用 step4 结果）

    Returns: dict with keys: accuracy, precision, recall, f1, avg_latency, file_path
    """
    outputs_dir = Path("outputs")
    if not outputs_dir.exists():
        return None

    result_files = list(outputs_dir.glob("**/inference_http_results_*.json"))
    if not result_files:
        return None

    # 按修改时间排序，取最新的
    latest_file = max(result_files, key=lambda f: f.stat().st_mtime)

    try:
        with open(latest_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        return {
            "file_path": str(latest_file),
            "accuracy": data.get("accuracy", 0),
            "precision": data.get("metrics", {}).get("precision", 0),
            "recall": data.get("metrics", {}).get("recall", 0),
            "f1": data.get("metrics", {}).get("f1", 0),
            "avg_latency": data.get("performance", {}).get("avg_latency_ms", 0),
            "total": data.get("total", 0),
            "correct": data.get("correct", 0),
            "model_path": data.get("model_path", ""),
            "engine": data.get("engine", ""),
        }
    except Exception as e:
        print(f"  [!] 解析结果文件失败: {e}")
        return None


def print_task_list(tasks: List[Tuple[str, str]]):
    """打印任务列表表格"""
    print(f"\n{'='*60}")
    print(f"【批量训练任务】")
    print(f"{'='*60}")
    print(f"任务数量: {len(tasks)}")
    print(f"{'─'*60}")
    print(f"{'序号':<4} {'模型':<35} {'提示词':<6} {'量化配置':<12}")
    print(f"{'─'*60}")

    for idx, (model, prompt_id) in enumerate(tasks, 1):
        quant_config = detect_quantization_config(model)
        if quant_config["quant_bits"]:
            quant_info = f"LoRA ({quant_config['quant_bits']}bit)"
        else:
            quant_info = "LoRA"
        model_short = model[:34]  # 截断过长的模型名
        print(f"  {idx:<2}  {model_short:<35} {prompt_id:<6} {quant_info:<12}")

    print(f"{'='*60}")


def print_comparison_report(task_results: List[Dict], log_dir: str = "outputs"):
    """
    打印对比分析报告（同时输出到控制台和日志文件）

    Args:
        task_results: list of dict, each contains:
            - idx: int
            - model: str
            - prompt_id: str
            - success: bool
            - metrics: dict or None
        log_dir: 日志输出目录
    """
    valid_results = [r for r in task_results if r["success"] and r["metrics"]]

    if not valid_results:
        print("\n[!] 没有有效的训练结果可用于对比")
        return

    # 创建日志文件
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    log_path = os.path.join(log_dir, f"model_compare_{timestamp}.log")

    # 定义输出函数（同时输出到控制台和文件）
    log_lines = []

    def log_print(msg: str):
        print(msg)
        log_lines.append(msg)

    log_print(f"\n{'='*70}")
    log_print(f"【模型效果对比分析】")
    log_print(f"{'='*70}")
    log_print(f"对比时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"数据来源: step4 HTTP API 测试结果")
    log_print(f"有效任务: {len(valid_results)} 个")

    # 打印对比表格
    log_print(f"\n{'任务':<6} {'模型':<20} {'提示词':<6} {'准确率':<8} {'精确率':<8} {'召回率':<8} {'F1':<8} {'平均延迟':<10}")
    log_print(f"{'─'*70}")

    for r in valid_results:
        m = r["metrics"]
        model_short = r["model"].split("/")[-1][:18]
        log_print(f"{r['idx']:<6} {model_short:<20} {r['prompt_id']:<6} "
              f"{m['accuracy']:<8.2f} {m['precision']:<8.2f} {m['recall']:<8.2f} "
              f"{m['f1']:<8.2f} {m['avg_latency']:<10.1f}")

    log_print(f"{'─'*70}")

    # 找出最佳模型
    best_by_accuracy = max(valid_results, key=lambda x: x["metrics"]["accuracy"])
    best_by_f1 = max(valid_results, key=lambda x: x["metrics"]["f1"])
    fastest = min(valid_results, key=lambda x: x["metrics"]["avg_latency"])

    log_print(f"\n【推荐】最佳准确率:")
    log_print(f"  任务 {best_by_accuracy['idx']}: {best_by_accuracy['model']} + 提示词 {best_by_accuracy['prompt_id']}")
    log_print(f"  准确率: {best_by_accuracy['metrics']['accuracy']:.2f}% | "
          f"F1: {best_by_accuracy['metrics']['f1']:.2f} | "
          f"测试用例: {best_by_accuracy['metrics']['correct']}/{best_by_accuracy['metrics']['total']}")

    if best_by_f1 != best_by_accuracy:
        log_print(f"\n【推荐】最佳 F1 分数:")
        log_print(f"  任务 {best_by_f1['idx']}: {best_by_f1['model']} + 提示词 {best_by_f1['prompt_id']}")
        log_print(f"  F1: {best_by_f1['metrics']['f1']:.2f} | 准确率: {best_by_f1['metrics']['accuracy']:.2f}%")

    log_print(f"\n【推荐】最快推理速度:")
    log_print(f"  任务 {fastest['idx']}: {fastest['model']} + 提示词 {fastest['prompt_id']}")
    log_print(f"  平均延迟: {fastest['metrics']['avg_latency']:.1f}ms | 准确率: {fastest['metrics']['accuracy']:.2f}%")

    # 如果有多个任务，给出综合推荐
    if len(valid_results) > 1:
        log_print(f"\n{'='*70}")
        log_print(f"【综合推荐】")

        # 计算综合得分（准确率权重 0.6，F1 权重 0.3，速度权重 0.1）
        max_latency = max(r["metrics"]["avg_latency"] for r in valid_results)
        for r in valid_results:
            acc_score = r["metrics"]["accuracy"]
            f1_score = r["metrics"]["f1"]
            speed_score = (1 - r["metrics"]["avg_latency"] / max_latency) * 100
            r["composite_score"] = acc_score * 0.6 + f1_score * 0.3 + speed_score * 0.1

        best_overall = max(valid_results, key=lambda x: x["composite_score"])
        log_print(f"  任务 {best_overall['idx']}: {best_overall['model']} + 提示词 {best_overall['prompt_id']}")
        log_print(f"  综合得分: {best_overall['composite_score']:.2f} (准确率 60% + F1 30% + 速度 10%)")
        log_print(f"  准确率: {best_overall['metrics']['accuracy']:.2f}% | "
              f"F1: {best_overall['metrics']['f1']:.2f} | "
              f"延迟: {best_overall['metrics']['avg_latency']:.1f}ms")

    log_print(f"{'='*70}\n")

    # 写入日志文件
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

    print(f"[*] 对比报告已保存: {log_path}")


# ============================================================
#  主流程
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="批量执行多个微调任务，完成后自动对比分析模型效果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认任务列表
  python run_pipeline_batch.py

  # 只测试量化模型
  python run_pipeline_batch.py --tasks "unsloth/Qwen3-4B-unsloth-bnb-4bit:001,unsloth/Qwen3-8B-unsloth-bnb-4bit:001"

  # 设置间隔时间
  python run_pipeline_batch.py --interval 10

  # 预览任务不执行
  python run_pipeline_batch.py --dry-run

量化配置自动检测:
  - 模型名包含 "-4bit" 或 "bnb-4bit" → QLoRA 4bit
  - 模型名包含 "-8bit" 或 "bnb-8bit" → QLoRA 8bit
  - 其他模型 → LoRA（非量化）
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"训练配置文件路径（默认: {DEFAULT_CONFIG_PATH}）"
    )

    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help="任务列表，格式: '模型1:prompt1,模型2:prompt2,...'"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=DEFAULT_INTERVAL_MINUTES,
        help=f"任务之间的间隔时间（分钟，默认: {DEFAULT_INTERVAL_MINUTES}）"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示将要执行的任务，不实际运行"
    )

    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="不等待，连续执行（无间隔）"
    )

    args = parser.parse_args()

    # 解析任务列表
    if args.tasks:
        tasks = parse_tasks_string(args.tasks)
        if not tasks:
            print("[!] 错误: 没有有效的任务")
            sys.exit(1)
    else:
        tasks = DEFAULT_TASKS

    # 显示任务列表
    print(f"配置文件: {args.config}")
    print(f"间隔时间: {args.interval} 分钟" if not args.no_wait else "间隔时间: 无等待")
    print_task_list(tasks)

    # Dry run 模式
    if args.dry_run:
        print("\n[*] Dry run 模式，不实际执行")
        sys.exit(0)

    # 确认
    response = input("\n是否开始执行？(y/n): ").strip().lower()
    if response != "y":
        print("[!] 已取消")
        sys.exit(0)

    # 备份原始配置
    config_path = args.config
    backup_path = config_path + ".batch_backup"
    shutil.copy2(config_path, backup_path)

    original_config = load_config(config_path)
    original_model = original_config["model"]["name_or_path"]
    original_prompt_id = original_config["data"].get("prompt_id")

    print(f"\n[*] 已备份原始配置文件: {backup_path}")
    print(f"[*] 原始配置: model={original_model}, prompt_id={original_prompt_id}")

    # 执行任务
    success_count = 0
    fail_count = 0
    start_time = datetime.now()
    task_results = []

    for idx, (model, prompt_id) in enumerate(tasks, 1):
        print(f"\n{'='*60}")
        print(f"【任务 {idx}/{len(tasks)}】")
        print(f"{'='*60}")
        print(f"模型: {model}")
        print(f"提示词ID: {prompt_id}")
        print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        task_success = False
        metrics = None

        try:
            # 更新配置（含自动量化检测）
            quant_config = update_config(config_path, model, prompt_id)
            if quant_config["quant_bits"]:
                quant_info = f"LoRA ({quant_config['quant_bits']}bit)"
            else:
                quant_info = "LoRA"
            print(f"[*] 已更新配置文件: model={model}, prompt_id={prompt_id}, 量化={quant_info}")

            # 执行训练
            if run_pipeline(prompt_id):
                print(f"[v] 任务 {idx} 完成")
                task_success = True
                success_count += 1

                # 收集结果
                print(f"[*] 正在收集训练结果...")
                metrics = find_latest_result_file(model, prompt_id)
                if metrics:
                    print(f"[*] 结果文件: {metrics['file_path']}")
                    print(f"[*] 准确率: {metrics['accuracy']:.2f}% | "
                          f"准确数: {metrics['correct']}/{metrics['total']}")
                else:
                    print(f"[!] 未找到推理结果文件")
            else:
                print(f"[x] 任务 {idx} 失败")
                fail_count += 1

        except Exception as e:
            print(f"[x] 任务 {idx} 异常: {e}")
            fail_count += 1

        # 记录结果
        task_results.append({
            "idx": idx,
            "model": model,
            "prompt_id": prompt_id,
            "success": task_success,
            "metrics": metrics,
        })

        # 等待（最后一个任务不等待）
        if idx < len(tasks) and not args.no_wait:
            wait_seconds = args.interval * 60
            print(f"\n[*] 等待 {args.interval} 分钟后开始下一个任务...")
            for remaining in range(wait_seconds, 0, -10):
                mins, secs = divmod(remaining, 60)
                print(f"  剩余: {mins:02d}:{secs:02d}", end="\r")
                time.sleep(10)
            print()

    # 恢复原始配置
    print(f"\n[*] 恢复原始配置...")
    shutil.copy2(backup_path, config_path)
    os.remove(backup_path)
    print(f"[v] 已恢复原始配置（保留格式和注释）")

    # 对比分析报告
    print_comparison_report(task_results, log_dir="outputs")

    # 总结
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"{'='*60}")
    print(f"【批量训练完成】")
    print(f"{'='*60}")
    print(f"成功: {success_count}/{len(tasks)}")
    print(f"失败: {fail_count}/{len(tasks)}")
    print(f"总耗时: {int(duration // 60)} 分 {int(duration % 60)} 秒")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
