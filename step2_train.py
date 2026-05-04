#!/usr/bin/env python3
"""
大厂标准 LLaMA Factory 微调训练脚本
支持: SFT / LoRA / QLoRA / Full Fine-tuning
版本: v1.0.0
"""

import os
import sys
import re
import glob
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
def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_ts = datetime.now().strftime("%m%d_%H%M")
    log_file = os.path.join(log_dir, f"step2_train_{log_ts}.log")

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


def generate_run_name(cfg: dict) -> str:
    """根据配置自动生成规范化 run_name: {task}_{model}_{lr}_{loraR}_{dataset}_{version}
    示例: majorintent_qwen8b_lr1e4_r8_intentcode_v1
    """
    # task
    task = cfg.get("run", {}).get("task", "majorintent")

    # model short name: "Qwen/Qwen3-8B" → "qwen8b"
    model_name = cfg.get("model", {}).get("name_or_path", "unknown")
    model_short = model_name.rstrip("/").split("/")[-1].lower()
    model_short = model_short.replace("-instruct", "")
    model_short = re.sub(r'(\w+?)[\d.]*-(\d+b)', r'\1\2', model_short)

    # lr: 1.0e-4 → "lr1e4"
    lr = cfg.get("training", {}).get("learning_rate", 1e-4)
    lr_str = f"{lr:.0e}"
    m = re.match(r'(\d+)e([+-])(\d+)', lr_str)
    if m:
        coef, sign, exp = m.groups()
        exp = exp.lstrip('0') or '0'
        lr_part = f"lr{coef}e{exp}"
    else:
        lr_part = f"lr{lr}"

    # lora rank
    lora_rank = cfg.get("lora", {}).get("rank", 8)
    lora_part = f"r{lora_rank}"

    # dataset: 自动检测 output 是编码还是中文名
    dataset = "intentcode"  # 默认
    data_dir = cfg.get("data", {}).get("dataset_dir", "data")
    dataset_name = cfg.get("data", {}).get("dataset_name", "")
    data_file = os.path.join(data_dir, f"{dataset_name}_train.jsonl")
    if os.path.exists(data_file):
        with open(data_file, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            if first_line:
                first_output = json.loads(first_line).get("output", "")
                if not re.match(r"^\d{3}$", first_output):
                    dataset = "intentname"

    # version
    version = cfg.get("run", {}).get("version", "v1")

    return f"{task}_{model_short}_{lr_part}_{lora_part}_{dataset}_{version}"


# ─── 模型检测与下载 ─────────────────────────────────────────────
def check_model_completeness(model_path: str, logger: logging.Logger) -> bool:
    """检查模型文件是否完整"""
    model_path = os.path.expanduser(model_path)

    # 处理 HuggingFace 模型 ID (如 "Qwen/Qwen3-8B")
    if "/" in model_path and not model_path.startswith("/"):
        # HuggingFace 缓存路径
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
        model_dir_name = model_path.replace("/", "--")
        possible_paths = [
            os.path.join(hf_cache, f"models--{model_dir_name}", "snapshots", "*"),
            os.path.join(hf_cache, model_path),
        ]

        actual_path = None
        for path in possible_paths:
            matches = Path(path).expanduser()
            if "*" in str(matches):
                import glob as _glob
                matches = _glob.glob(str(matches))
                if matches:
                    actual_path = matches[0]
                    break
            elif matches.exists():
                actual_path = str(matches)
                break

        if actual_path is None:
            logger.info(f"  模型未找到: {model_path}")
            return False

        model_path = actual_path
    elif not os.path.isabs(model_path):
        model_path = os.path.abspath(model_path)

    logger.info(f"  检查路径: {model_path}")

    # 检查必需文件
    required_files = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    missing_files = []

    for file in required_files:
        file_path = os.path.join(model_path, file)
        if not os.path.exists(file_path):
            missing_files.append(file)

    # 检查 safetensors 权重文件
    index_file = os.path.join(model_path, "model.safetensors.index.json")
    has_weights = False

    if os.path.exists(index_file):
        try:
            with open(index_file, "r") as f:
                index = json.load(f)
            total_size = index.get("metadata", {}).get("total_size", 0)
            weight_map = index.get("weight_map", {})

            weight_files = set(weight_map.values())
            missing_weights = []
            for wf in weight_files:
                wf_path = os.path.join(model_path, wf)
                if not os.path.exists(wf_path):
                    missing_weights.append(wf)

            if missing_weights:
                logger.warning(f"  缺少权重文件: {missing_weights}")
            else:
                has_weights = True
                logger.info(f"  权重文件完整 ({total_size / 1e9:.2f} GB)")
        except Exception as e:
            logger.warning(f"  读取 safetensors 索引失败: {e}")
    else:
        safetensors_files = list(Path(model_path).glob("*.safetensors"))
        bin_files = list(Path(model_path).glob("*.bin"))

        if safetensors_files:
            total_size = sum(f.stat().st_size for f in safetensors_files)
            logger.info(f"  找到 {len(safetensors_files)} 个 safetensors 文件 ({total_size / 1e9:.2f} GB)")
            has_weights = True
        elif bin_files:
            total_size = sum(f.stat().st_size for f in bin_files)
            logger.info(f"  找到 {len(bin_files)} 个 bin 文件 ({total_size / 1e9:.2f} GB)")
            has_weights = True
        else:
            logger.warning("  未找到模型权重文件")

    if missing_files:
        logger.warning(f"  缺少配置文件: {missing_files}")
        return False

    if not has_weights:
        logger.warning("  模型权重不完整")
        return False

    logger.info("  模型完整性检查通过")
    return True


def download_model_with_modelscope(
    model_id: str,
    cache_dir: str,
    logger: logging.Logger
) -> str:
    """使用 ModelScope 下载模型到 HuggingFace 缓存路径"""
    logger.info(f"  使用 ModelScope 下载: {model_id}")

    try:
        from modelscope import snapshot_download
    except ImportError:
        logger.error("  ModelScope 未安装，正在安装...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "modelscope", "-q"],
            check=True
        )
        from modelscope import snapshot_download
        logger.info("  ModelScope 安装完成")

    os.makedirs(cache_dir, exist_ok=True)

    import logging as ms_logging
    ms_logging.basicConfig(level=logging.INFO)

    logger.info("  开始下载...")
    model_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        revision='master'
    )

    logger.info(f"  下载完成: {model_path}")
    return model_path


def ensure_model_available(cfg: dict, logger: logging.Logger) -> str:
    """确保模型可用：依次检查本地路径、HuggingFace 缓存、ModelScope 缓存，都不存在则下载"""
    model_name_or_path = cfg["model"]["name_or_path"]

    logger.info(f"{'─'*60}")
    logger.info(f"  【模型检查】")
    logger.info(f"{'─'*60}")
    logger.info(f"  模型: {model_name_or_path}")

    # 1. 本地路径
    if os.path.exists(model_name_or_path):
        logger.info(f"  使用本地模型: {model_name_or_path}")
        if check_model_completeness(model_name_or_path, logger):
            return model_name_or_path
        else:
            raise FileNotFoundError(f"本地模型不完整: {model_name_or_path}")

    # 2. HuggingFace 缓存
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir_name = model_name_or_path.replace("/", "--")
    hf_model_path = os.path.join(hf_cache, f"models--{model_dir_name}")

    snapshot_dirs = glob.glob(os.path.join(hf_model_path, "snapshots", "*"))
    if snapshot_dirs:
        logger.info(f"  检查 HuggingFace 缓存: {snapshot_dirs[0]}")
        if check_model_completeness(model_name_or_path, logger):
            logger.info(f"  使用 HuggingFace 缓存: {snapshot_dirs[0]}")
            cfg["model"]["name_or_path"] = snapshot_dirs[0]
            return snapshot_dirs[0]

    # 3. ModelScope 缓存
    ms_cache_paths = [
        os.path.expanduser("~/.cache/modelscope/hub"),
        os.path.expanduser("~/.modelscope/hub"),
    ]
    for ms_cache in ms_cache_paths:
        ms_model_path = os.path.join(ms_cache, model_name_or_path)
        if os.path.exists(ms_model_path):
            logger.info(f"  检查 ModelScope 缓存: {ms_model_path}")
            if check_model_completeness(ms_model_path, logger):
                logger.info(f"  使用 ModelScope 缓存: {ms_model_path}")
                cfg["model"]["name_or_path"] = ms_model_path
                return ms_model_path

    # 4. 缓存中均未找到，通过 ModelScope 下载
    logger.warning(f"  本地未找到模型，将通过 ModelScope 下载")

    modelscope_id_map = {
        "Qwen/Qwen3-8B": "Qwen/Qwen3-8B",
        "Qwen/Qwen3-4B": "Qwen/Qwen3-4B",
        "Qwen/Qwen3-1.7B": "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3.5-9B": "Qwen/Qwen3.5-9B",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "AI-ModelScope/Meta-Llama-3.1-8B-Instruct",
    }
    modelscope_id = modelscope_id_map.get(model_name_or_path, model_name_or_path)

    try:
        downloaded_path = download_model_with_modelscope(
            modelscope_id,
            hf_cache,
            logger
        )
        cfg["model"]["name_or_path"] = downloaded_path
        logger.info(f"  模型已下载: {downloaded_path}")
        return downloaded_path

    except Exception as e:
        logger.error(f"  下载失败: {e}")
        raise RuntimeError(f"无法下载模型 {model_name_or_path}: {e}")


# ─── 环境检查 ────────────────────────────────────────────────
def check_environment(logger: logging.Logger) -> dict:
    logger.info(f"{'─'*60}")
    logger.info(f"  【环境检查】")
    logger.info(f"{'─'*60}")

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
    disk = shutil.disk_usage(os.getcwd())
    free_gb = disk.free / 1e9
    env_info["disk_free_gb"] = round(free_gb, 1)
    logger.info(f"  磁盘剩余: {free_gb:.1f} GB")
    if free_gb < 50:
        logger.warning("  磁盘剩余不足 50GB，建议清理后再训练")

    return env_info


# ─── 数据验证 ────────────────────────────────────────────────
def validate_dataset(data_path: str, logger: logging.Logger) -> dict:
    logger.info(f"{'─'*60}")
    logger.info(f"  【数据集验证】")
    logger.info(f"{'─'*60}")

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

    # 字段检查（验证全部数据）
    for i, rec in enumerate(records):
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
            logger.warning(f"  {err}")

    if stats["invalid"] / max(stats["total"], 1) > 0.05:
        raise ValueError(f"数据集异常比例 > 5%，请检查数据质量！")

    logger.info("  数据集验证通过")

    # 打印数据样例
    logger.info(f"  {'─'*40}")
    logger.info(f"  数据样例（前 3 条）:")
    for i, rec in enumerate(records[:3], 1):
        logger.info(f"  [{i}] {json.dumps(rec, ensure_ascii=False)}")

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
        "val_size": cfg["training"].get("val_size", 0.2),
        "eval_strategy": cfg["training"].get("eval_strategy", "steps"),
        "eval_steps": cfg["training"].get("eval_steps", 100),
        "load_best_model_at_end": False,
        "save_total_limit": cfg["training"].get("save_total_limit", 5),

        # 量化（QLoRA）
        "quantization_bit": cfg["quantization"].get("bits", None) if cfg.get("quantization", {}).get("enable") else None,

        # 日志
        "logging_steps": cfg["training"].get("logging_steps", 10),
        "report_to": cfg["logging"].get("report_to", "tensorboard"),
        "run_name": run_name,

        # Flash Attention
        "flash_attn": cfg["training"].get("flash_attention", "sdpa"),
    }

    ft_type = cfg["training"].get("finetuning_type", "lora")
    if ft_type in ("lora", "qlora"):
        lf_config["save_only_model"] = True

    if cfg["training"].get("seed") is not None:
        lf_config["seed"] = cfg["training"]["seed"]

    if cfg["training"].get("gradient_checkpointing", False):
        lf_config["gradient_checkpointing"] = True

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
    logger.info(f"{'─'*60}")
    logger.info(f"  【启动训练】")
    logger.info(f"{'─'*60}")

    cmd = ["llamafactory-cli", "train", lf_config_path]
    logger.info(f"  训练命令: {' '.join(cmd)}")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

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


# ─── 智能选择最佳 Checkpoint ─────────────────────────────────────
def evaluate_all_checkpoints(
    lf_config_path: str,
    output_dir: str,
    logger: logging.Logger
) -> str:
    """训练结束后评估所有 checkpoint，返回 loss 最小的那个"""
    logger.info(f"{'─'*60}")
    logger.info(f"  【评估所有 Checkpoint】")
    logger.info(f"{'─'*60}")

    # 收集所有 checkpoint
    checkpoints = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
                checkpoints.append((step, full))
            except (ValueError, IndexError):
                pass

    if not checkpoints:
        raise FileNotFoundError(f"未找到 checkpoint 目录: {output_dir}")

    checkpoints.sort(key=lambda x: x[0])
    logger.info(f"  找到 {len(checkpoints)} 个 checkpoint: {[c[0] for c in checkpoints]}")

    # 读取训练配置
    with open(lf_config_path, "r", encoding="utf-8") as f:
        train_config = yaml.safe_load(f)

    train_config["load_best_model_at_end"] = False
    train_config["do_train"] = False
    train_config["do_predict"] = False

    eval_results = {}

    for step, ckpt_path in checkpoints:
        logger.info(f"  评估 checkpoint-{step}...")
        train_config["model_name_or_path"] = train_config.get("model_name_or_path")
        train_config["adapter_name_or_path"] = ckpt_path
        train_config["output_dir"] = os.path.join(output_dir, f"eval_{step}")

        eval_config_path = os.path.join(output_dir, f"eval_{step}.yaml")
        with open(eval_config_path, "w", encoding="utf-8") as f:
            yaml.dump(train_config, f, allow_unicode=True, sort_keys=False)

        result = subprocess.run(
            ["llamafactory-cli", "eval", eval_config_path],
            capture_output=True, text=True, timeout=600
        )

        if result.returncode == 0:
            output = result.stdout + result.stderr
            loss_match = re.search(r"eval_loss[:\s]+([0-9.]+)", output)
            if loss_match:
                loss = float(loss_match.group(1))
                eval_results[step] = loss
                logger.info(f"    checkpoint-{step}: loss = {loss:.4f}")
            else:
                logger.warning(f"    checkpoint-{step}: 无法解析 loss")
        else:
            logger.warning(f"    checkpoint-{step}: 评估失败")

        try:
            os.remove(eval_config_path)
        except:
            pass

    if eval_results:
        best_step = min(eval_results, key=eval_results.get)
        best_loss = eval_results[best_step]
        best_path = os.path.join(output_dir, f"checkpoint-{best_step}")
        logger.info(f"  最佳 checkpoint: {best_step} (loss={best_loss:.4f})")
        return best_path

    checkpoints.sort(key=lambda x: x[0], reverse=True)
    logger.warning("  评估失败，使用最新 checkpoint")
    return checkpoints[0][1]


# ─── 查找最佳 Checkpoint ──────────────────────────────────────
def find_best_checkpoint(
    output_dir: str,
    logger: logging.Logger,
    lf_config_path: Optional[str] = None
) -> str:
    """在 output_dir 下查找 trainer_state.json 中记录的最佳 checkpoint"""
    state_path = os.path.join(output_dir, "trainer_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        best = state.get("best_model_checkpoint")
        if best and os.path.isdir(best):
            logger.info(f"  trainer_state.json 记录的最佳 checkpoint: {best}")
            if lf_config_path:
                logger.info(f"  执行最终评估以确认最佳 checkpoint...")
                return evaluate_all_checkpoints(lf_config_path, output_dir, logger)
            return best

    # 回退：选择编号最大的 checkpoint 目录
    checkpoints = []
    for name in os.listdir(output_dir):
        full = os.path.join(output_dir, name)
        if os.path.isdir(full) and name.startswith("checkpoint-"):
            try:
                step = int(name.split("-")[1])
                checkpoints.append((step, full))
            except (ValueError, IndexError):
                pass
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0], reverse=True)
        best = checkpoints[0][1]
        logger.info(f"  使用最新 checkpoint: {best}")
        return best

    raise FileNotFoundError(f"未找到 checkpoint 目录: {output_dir}")


# ─── 模型合并（LoRA → 完整模型）────────────────────────────
def merge_lora_weights(cfg: dict, adapter_path: str, output_path: str, logger: logging.Logger):
    logger.info(f"{'─'*60}")
    logger.info(f"  【合并 LoRA 权重】")
    logger.info(f"{'─'*60}")

    merge_config = {
        "model_name_or_path": cfg["model"]["name_or_path"],
        "adapter_name_or_path": adapter_path,
        "template": cfg["data"].get("template", "llama3"),
        "finetuning_type": "lora",
        "export_dir": output_path,
        "export_size": 4,
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

    logger.info(f"  合并完成，保存至: {output_path}")


# ─── 评估 ────────────────────────────────────────────────────
def run_evaluation(cfg: dict, model_path: str, logger: logging.Logger) -> dict:
    logger.info(f"{'─'*60}")
    logger.info(f"  【模型评估】")
    logger.info(f"{'─'*60}")

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
            results_file = os.path.join(model_path, "eval_results", task, "results.json")
            if os.path.exists(results_file):
                with open(results_file) as f:
                    results[task] = json.load(f)
                logger.info(f"  {task}: {results[task]}")
            else:
                logger.warning(f"  {task} 结果文件不存在")
        else:
            logger.warning(f"  {task} 评估失败: {result.stderr[:200]}")

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
    logger.info(f"{'─'*60}")
    logger.info(f"  【保存训练报告】")
    logger.info(f"{'─'*60}")

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

    summary_path = os.path.join(output_dir, "run_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*60}\n")
        f.write(f"训练运行摘要: {run_name}\n")
        f.write(f"{'='*60}\n")
        f.write(f"状态:     {'成功' if success else '失败'}\n")
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

    logger.info(f"  报告已保存: {report_path}")
    logger.info(f"  摘要已保存: {summary_path}")


# ─── 主流程 ────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="LLaMA Factory 大厂标准微调脚本")
    parser.add_argument("--config", type=str, default="config/train_config.yaml", help="训练配置文件路径 (yaml/json)，默认: config/train_config.yaml")
    parser.add_argument("--run_name", type=str, default=None, help="运行名称（默认自动生成）")
    parser.add_argument("--skip_merge", action="store_true", help="跳过 LoRA 权重合并")
    parser.add_argument("--skip_eval", action="store_true", help="跳过模型评估")
    parser.add_argument("--dry_run", action="store_true", help="仅验证配置，不实际训练")
    args = parser.parse_args()

    # 加载配置
    cfg = load_config(args.config)

    # 运行名称：命令行指定 > 复用已有 run 目录 > 新建（基础名 + 日期时分）
    base_output = cfg.get("output", {}).get("base_dir", "outputs")
    if args.run_name:
        run_name = args.run_name
    else:
        base_name = generate_run_name(cfg)
        existing = sorted(
            glob.glob(os.path.join(base_output, f"{base_name}_*")),
            key=os.path.getmtime, reverse=True
        )
        if existing and os.path.isdir(existing[0]):
            run_name = os.path.basename(existing[0])
        else:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            run_name = f"{base_name}_{timestamp}"

    # 输出目录
    output_dir = os.path.join(base_output, run_name)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(output_dir, exist_ok=True)

    # 日志
    logger = setup_logger(log_dir)

    logger.info(f"{'='*60}")
    logger.info(f"  Step 2 — 微调训练")
    logger.info(f"{'='*60}")
    logger.info(f"  运行名称 : {run_name}")
    logger.info(f"  配置文件 : {args.config}")
    logger.info(f"  输出目录 : {output_dir}")

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

        # 2. 模型检查与下载
        model_path = ensure_model_available(cfg, logger)

        # 3. 数据验证
        data_path = os.path.join(
            cfg["data"].get("dataset_dir", "data"),
            f"{cfg['data']['dataset_name']}_train.jsonl"
        )
        if os.path.exists(data_path):
            data_stats = validate_dataset(data_path, logger)
        else:
            logger.warning(f"  数据文件 {data_path} 不存在，跳过验证（将由 LLaMA Factory 内部处理）")

        if args.dry_run:
            logger.info(f"{'─'*60}")
            logger.info(f"  Dry Run 模式 — 验证通过，跳过实际训练")
            logger.info(f"{'─'*60}")
            success = True
            return

        # 4. 生成训练配置
        lf_config_path = generate_llamafactory_config(cfg, run_name, output_dir)
        logger.info(f"  LLaMA Factory 配置已生成: {lf_config_path}")

        # 5. 执行训练
        process = run_training(lf_config_path, logger)
        if process.returncode != 0:
            raise RuntimeError(f"训练进程退出码: {process.returncode}")
        logger.info("  训练完成")

        # 6. 合并 LoRA 权重
        # 检测是否为预量化模型（Unsloth 等），预量化模型无法合并
        model_name = cfg["model"].get("name_or_path", "")
        is_preqantized = (
            "unsloth" in model_name.lower() or
            "-bnb-" in model_name.lower() or
            "-awq" in model_name.lower() or
            "-gptq" in model_name.lower()
        )

        if not args.skip_merge and cfg["training"].get("finetuning_type") in ("lora", "qlora"):
            if is_preqantized:
                logger.info(f"{'─'*60}")
                logger.info(f"  【跳过合并】检测到预量化模型: {model_name}")
                logger.info(f"  预量化模型的 LoRA 权重无法合并，将直接使用 checkpoint")
                logger.info(f"{'─'*60}")
                eval_model_path = output_dir
            else:
                best_ckpt = find_best_checkpoint(output_dir, logger, lf_config_path)
                merged_path = os.path.join(output_dir, "merged_model")
                merge_lora_weights(cfg, best_ckpt, merged_path, logger)
                eval_model_path = merged_path
        else:
            eval_model_path = output_dir

        # 7. 模型评估
        if not args.skip_eval:
            eval_results = run_evaluation(cfg, eval_model_path, logger)

        success = True
        logger.info(f"{'='*60}")
        logger.info(f"  训练完成: {run_name}")
        logger.info(f"{'='*60}")

    except Exception as e:
        logger.error(f"{'='*60}")
        logger.error(f"  训练失败: {e}", exc_info=True)
        logger.error(f"{'='*60}")

    finally:
        save_run_report(
            run_name, cfg, env_info,
            data_stats, eval_results, output_dir,
            start_time, success, logger
        )


if __name__ == "__main__":
    main()
