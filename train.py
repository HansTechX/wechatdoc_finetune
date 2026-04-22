#!/usr/bin/env python3
"""
大厂标准 LLaMA Factory 微调训练脚本
支持: SFT / LoRA / QLoRA / Full Fine-tuning
版本: v1.0.0
"""

import os
import sys
import json
import time
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional

import yaml

# ─── 日志配置 ────────────────────────────────────────────────
def setup_logger(log_dir: str, run_name: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{run_name}.log")

    logger = logging.getLogger("finetune")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 控制台 Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # 文件 Handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ─── 配置加载 ────────────────────────────────────────────────
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        if config_path.endswith(".yaml") or config_path.endswith(".yml"):
            return yaml.safe_load(f)
        elif config_path.endswith(".json"):
            return json.load(f)
    raise ValueError(f"不支持的配置格式: {config_path}")


# ─── 环境检查 ────────────────────────────────────────────────
def check_environment(logger: logging.Logger) -> dict:
    logger.info("=" * 60)
    logger.info("【环境检查】")

    env_info = {}

    # Python 版本
    env_info["python"] = sys.version
    logger.info(f"  Python: {sys.version.split()[0]}")

    # CUDA / GPU
    try:
        import torch
        env_info["torch"] = torch.__version__
        env_info["cuda_available"] = torch.cuda.is_available()
        env_info["gpu_count"] = torch.cuda.device_count()
        logger.info(f"  PyTorch: {torch.__version__}")
        logger.info(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(f"  GPU[{i}]: {gpu_name} ({gpu_mem:.1f} GB)")
                env_info[f"gpu_{i}"] = {"name": gpu_name, "memory_gb": round(gpu_mem, 1)}
    except ImportError:
        logger.warning("  未安装 PyTorch，请先安装依赖")
        env_info["torch"] = None

    # LLaMA Factory 版本
    try:
        result = subprocess.run(
            ["python", "-c", "import llamafactory; print(llamafactory.__version__)"],
            capture_output=True, text=True
        )
        version = result.stdout.strip()
        env_info["llamafactory"] = version
        logger.info(f"  LLaMA Factory: {version}")
    except Exception:
        logger.warning("  LLaMA Factory 未安装，请执行: pip install llamafactory")
        env_info["llamafactory"] = None

    # 磁盘空间
    disk = shutil.disk_usage(os.path.dirname(os.path.abspath(__file__)))
    free_gb = disk.free / 1e9
    env_info["disk_free_gb"] = round(free_gb, 1)
    logger.info(f"  磁盘剩余: {free_gb:.1f} GB")
    if free_gb < 50:
        logger.warning("  [!] 磁盘剩余不足 50GB，建议清理后再训练")

    logger.info("=" * 60)
    return env_info


# ─── 数据验证 ────────────────────────────────────────────────
def validate_dataset(data_path: str, logger: logging.Logger) -> dict:
    logger.info("【数据集验证】")
    stats = {"total": 0, "valid": 0, "invalid": 0, "errors": []}

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集路径不存在: {data_path}")

    # 支持 .json / .jsonl
    if data_path.endswith(".jsonl"):
        with open(data_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        records = []
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                stats["errors"].append(f"第 {i+1} 行 JSON 解析失败: {e}")
    else:
        with open(data_path, "r", encoding="utf-8") as f:
            records = json.load(f)

    stats["total"] = len(records)

    # 字段检查（Alpaca / ShareGPT 格式）
    for i, rec in enumerate(records[:5000]):  # 最多抽查 5000 条
        if "conversations" in rec:  # ShareGPT 格式
            valid = (
                isinstance(rec.get("conversations"), list) and
                len(rec["conversations"]) >= 2
            )
        elif "instruction" in rec:  # Alpaca 格式
            valid = bool(rec.get("instruction", "").strip())
        else:
            valid = False
            stats["errors"].append(f"第 {i+1} 条记录格式不识别，需包含 'conversations' 或 'instruction' 字段")

        if valid:
            stats["valid"] += 1
        else:
            stats["invalid"] += 1

    logger.info(f"  总条数: {stats['total']}")
    logger.info(f"  格式正确: {stats['valid']}")
    logger.info(f"  格式异常: {stats['invalid']}")
    if stats["errors"]:
        for err in stats["errors"][:5]:
            logger.warning(f"  [!] {err}")

    if stats["invalid"] / max(stats["total"], 1) > 0.05:
        raise ValueError(f"数据集异常比例 > 5%，请检查数据质量！")

    logger.info("  [v] 数据集验证通过")
    return stats


# ─── 生成 LLaMA Factory 训练配置 ────────────────────────────
def generate_llamafactory_config(cfg: dict, run_name: str, output_dir: str) -> str:
    """将通用配置转换为 LLaMA Factory YAML 格式"""

    lf_config = {
        # 模型
        "model_name_or_path": cfg["model"]["name_or_path"],
        "trust_remote_code": cfg["model"].get("trust_remote_code", True),

        # 训练方式
        "stage": cfg["training"].get("stage", "sft"),
        "do_train": True,
        "finetuning_type": cfg["training"].get("finetuning_type", "lora"),

        # LoRA 配置
        "lora_rank": cfg["lora"].get("rank", 8),
        "lora_alpha": cfg["lora"].get("alpha", 16),
        "lora_dropout": cfg["lora"].get("dropout", 0.05),
        "lora_target": cfg["lora"].get("target_modules", "all"),

        # 数据集
        "dataset": cfg["data"]["dataset_name"],
        "dataset_dir": cfg["data"].get("dataset_dir", "data"),
        "template": cfg["data"].get("template", "llama3"),
        "cutoff_len": cfg["data"].get("max_seq_length", 2048),
        "max_samples": cfg["data"].get("max_samples", None),
        "overwrite_cache": True,
        "preprocessing_num_workers": cfg["data"].get("num_workers", 4),

        # 输出
        "output_dir": output_dir,
        "overwrite_output_dir": True,
        "save_strategy": cfg["training"].get("save_strategy", "steps"),
        "save_steps": cfg["training"].get("save_steps", 100),
        "save_total_limit": cfg["training"].get("save_total_limit", 3),

        # 训练超参数
        "per_device_train_batch_size": cfg["training"].get("per_device_train_batch_size", 2),
        "gradient_accumulation_steps": cfg["training"].get("gradient_accumulation_steps", 4),
        "learning_rate": cfg["training"].get("learning_rate", 1e-4),
        "num_train_epochs": cfg["training"].get("num_epochs", 3),
        "lr_scheduler_type": cfg["training"].get("lr_scheduler", "cosine"),
        "warmup_ratio": cfg["training"].get("warmup_ratio", 0.05),
        "weight_decay": cfg["training"].get("weight_decay", 0.01),
        "max_grad_norm": cfg["training"].get("max_grad_norm", 1.0),
        "bf16": cfg["training"].get("bf16", True),
        "fp16": cfg["training"].get("fp16", False),

        # 评估
        "val_size": cfg["training"].get("val_size", 0.01),
        "evaluation_strategy": cfg["training"].get("eval_strategy", "steps"),
        "eval_steps": cfg["training"].get("eval_steps", 100),
        "load_best_model_at_end": True,

        # 量化（QLoRA）
        "quantization_bit": cfg["quantization"].get("bits", None) if cfg.get("quantization", {}).get("enable") else None,

        # 日志
        "logging_steps": cfg["training"].get("logging_steps", 10),
        "report_to": cfg["logging"].get("report_to", "tensorboard"),
        "run_name": run_name,

        # Flash Attention
        "flash_attn": cfg["training"].get("flash_attention", "fa2"),
    }

    # 清理 None 值
    lf_config = {k: v for k, v in lf_config.items() if v is not None}

    config_path = os.path.join(output_dir, "llamafactory_train.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(lf_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    return config_path


# ─── 执行训练 ────────────────────────────────────────────────
def run_training(
    lf_config_path: str,
    logger: logging.Logger
) -> subprocess.CompletedProcess:
    logger.info("【启动训练】")

    cmd = ["llamafactory-cli", "train", lf_config_path]
    logger.info(f"  训练命令: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # 实时流式输出
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1
    )

    for line in process.stdout:
        line = line.rstrip()
        if line:
            logger.info(f"  [trainer] {line}")

    process.wait()
    return process


# ─── 模型合并（LoRA → 完整模型）────────────────────────────
def merge_lora_weights(cfg: dict, adapter_path: str, output_path: str, logger: logging.Logger):
    logger.info("【合并 LoRA 权重】")

    merge_config = {
        "model_name_or_path": cfg["model"]["name_or_path"],
        "adapter_name_or_path": adapter_path,
        "template": cfg["data"].get("template", "llama3"),
        "finetuning_type": "lora",
        "export_dir": output_path,
        "export_size": 4,  # 每个分片 4GB
        "export_legacy_format": False,
    }

    merge_config_path = os.path.join(output_path, "merge_config.yaml")
    os.makedirs(output_path, exist_ok=True)
    with open(merge_config_path, "w") as f:
        yaml.dump(merge_config, f, allow_unicode=True)

    cmd = ["llamafactory-cli", "export", merge_config_path]
    logger.info(f"  合并命令: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"  合并失败:\n{result.stderr}")
        raise RuntimeError("LoRA 权重合并失败")

    logger.info(f"  [v] 合并完成，保存至: {output_path}")


# ─── 评估 ────────────────────────────────────────────────────
def run_evaluation(cfg: dict, model_path: str, logger: logging.Logger) -> dict:
    logger.info("【模型评估】")

    eval_tasks = cfg.get("evaluation", {}).get("tasks", [])
    if not eval_tasks:
        logger.info("  未配置评估任务，跳过")
        return {}

    results = {}

    for task in eval_tasks:
        logger.info(f"  评估任务: {task}")
        eval_config = {
            "model_name_or_path": model_path,
            "template": cfg["data"].get("template", "llama3"),
            "task": task,
            "split": "test",
            "lang": cfg.get("evaluation", {}).get("lang", "zh"),
            "n_shot": cfg.get("evaluation", {}).get("n_shot", 5),
            "batch_size": cfg.get("evaluation", {}).get("batch_size", 4),
            "save_dir": os.path.join(model_path, "eval_results", task),
        }

        eval_config_path = os.path.join(model_path, f"eval_{task}.yaml")
        with open(eval_config_path, "w") as f:
            yaml.dump(eval_config, f, allow_unicode=True)

        result = subprocess.run(
            ["llamafactory-cli", "eval", eval_config_path],
            capture_output=True, text=True
        )

        if result.returncode == 0:
            # 尝试解析评估结果
            results_file = os.path.join(model_path, "eval_results", task, "results.json")
            if os.path.exists(results_file):
                with open(results_file) as f:
                    results[task] = json.load(f)
                logger.info(f"  [v] {task}: {results[task]}")
            else:
                logger.warning(f"  [!] {task} 结果文件不存在")
        else:
            logger.warning(f"  [!] {task} 评估失败: {result.stderr[:200]}")

    return results


# ─── 训练结果记录 ─────────────────────────────────────────────
def save_run_report(
    run_name: str,
    cfg: dict,
    env_info: dict,
    data_stats: dict,
    eval_results: dict,
    output_dir: str,
    start_time: float,
    success: bool,
    logger: logging.Logger
):
    logger.info("【保存训练报告】")

    duration = time.time() - start_time
    hours, rem = divmod(int(duration), 3600)
    minutes, seconds = divmod(rem, 60)

    report = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "duration": f"{hours:02d}h {minutes:02d}m {seconds:02d}s",
        "duration_seconds": round(duration, 1),
        "status": "success" if success else "failed",
        "environment": env_info,
        "config": cfg,
        "data_stats": data_stats,
        "eval_results": eval_results,
        "output_dir": output_dir,
    }

    report_path = os.path.join(output_dir, "run_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 同时生成人类可读的摘要
    summary_path = os.path.join(output_dir, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"训练运行摘要: {run_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"状态:     {'[v] 成功' if success else '[x] 失败'}\n")
        f.write(f"时间:     {report['timestamp']}\n")
        f.write(f"耗时:     {report['duration']}\n")
        f.write(f"模型:     {cfg['model']['name_or_path']}\n")
        f.write(f"训练方式: {cfg['training'].get('finetuning_type', 'lora')}\n")
        f.write(f"数据集:   {cfg['data']['dataset_name']} ({data_stats.get('total', '?')} 条)\n")
        f.write(f"Epochs:  {cfg['training'].get('num_epochs', '?')}\n")
        f.write(f"LR:      {cfg['training'].get('learning_rate', '?')}\n")
        f.write(f"\n【评估结果】\n")
        if eval_results:
            for task, result in eval_results.items():
                f.write(f"  {task}: {result}\n")
        else:
            f.write("  （未配置评估任务）\n")
        f.write(f"\n输出目录: {output_dir}\n")
        f.write(f"{'='*60}\n")

    logger.info(f"  [v] 报告已保存: {report_path}")
    logger.info(f"  [v] 摘要已保存: {summary_path}")


# ─── 主流程 ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LLaMA Factory 大厂标准微调脚本")
    parser.add_argument("--config", type=str, required=True, help="训练配置文件路径 (yaml/json)")
    parser.add_argument("--run_name", type=str, default=None, help="运行名称（默认自动生成）")
    parser.add_argument("--skip_merge", action="store_true", help="跳过 LoRA 权重合并")
    parser.add_argument("--skip_eval", action="store_true", help="跳过模型评估")
    parser.add_argument("--dry_run", action="store_true", help="仅验证配置，不实际训练")
    args = parser.parse_args()

    # 运行名称
    if args.run_name:
        run_name = args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"

    # 加载配置
    cfg = load_config(args.config)

    # 输出目录
    base_output = cfg.get("output", {}).get("base_dir", "outputs")
    output_dir = os.path.join(base_output, run_name)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)

    # 日志
    logger = setup_logger(log_dir, run_name)
    logger.info(f"[>] 开始训练任务: {run_name}")
    logger.info(f"   配置文件: {args.config}")
    logger.info(f"   输出目录: {output_dir}")

    # 备份配置
    shutil.copy2(args.config, os.path.join(output_dir, "config_backup.yaml"))

    start_time = time.time()
    success = False
    data_stats = {}
    eval_results = {}
    env_info = {}

    try:
        # 1. 环境检查
        env_info = check_environment(logger)

        # 2. 数据验证
        data_path = os.path.join(
            cfg["data"].get("dataset_dir", "data"),
            cfg["data"]["dataset_name"] + ".jsonl"
        )
        if os.path.exists(data_path):
            data_stats = validate_dataset(data_path, logger)
        else:
            logger.warning(f"  数据文件 {data_path} 不存在，跳过验证（将由 LLaMA Factory 内部处理）")

        if args.dry_run:
            logger.info("【Dry Run 模式】验证通过，跳过实际训练")
            success = True
            return

        # 3. 生成训练配置
        lf_config_path = generate_llamafactory_config(cfg, run_name, output_dir)
        logger.info(f"  [v] LLaMA Factory 配置已生成: {lf_config_path}")

        # 4. 执行训练
        process = run_training(lf_config_path, logger)
        if process.returncode != 0:
            raise RuntimeError(f"训练进程退出码: {process.returncode}")
        logger.info("  [v] 训练完成")

        # 5. 合并 LoRA 权重
        if not args.skip_merge and cfg["training"].get("finetuning_type") == "lora":
            merged_path = os.path.join(output_dir, "merged_model")
            merge_lora_weights(cfg, output_dir, merged_path, logger)
            eval_model_path = merged_path
        else:
            eval_model_path = output_dir

        # 6. 模型评估
        if not args.skip_eval:
            eval_results = run_evaluation(cfg, eval_model_path, logger)

        success = True
        logger.info(f"\n{'='*60}")
        logger.info(f"[v] 训练任务 [{run_name}] 完成！")
        logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"\n{'='*60}")
        logger.error(f"[x] 训练任务失败: {e}", exc_info=True)
        logger.error(f"{'='*60}")

    finally:
        # 7. 保存报告
        save_run_report(
            run_name, cfg, env_info,
            data_stats, eval_results, output_dir,
            start_time, success, logger
        )


if __name__ == "__main__":
    main()
