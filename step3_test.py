#!/usr/bin/env python3
"""
模型推理测试脚本 - 意图分类专用（HuggingFace Transformers 版）
基于 Qwen3 模型，使用标准 system + user 消息格式
训练完成后，用于验证模型的意图识别准确率
"""

import os
import sys
import re
import glob
import json
import logging
import argparse
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from step2_train import generate_run_name, load_config


# ─── 日志配置 ────────────────────────────────────────────────
logger = logging.getLogger("step3_test")
detail_logger = logging.getLogger("step3_test_detail")


def setup_logger(log_dir: str) -> tuple:
    """配置日志：main (控制台+文件) + detail (仅文件)
    Returns: (main_log_path, detail_log_path)
    """
    log_ts = datetime.now().strftime("%m%d_%H%M")
    main_log_path = os.path.join(log_dir, f"step3_test_{log_ts}.log")
    detail_log_path = os.path.join(log_dir, f"step3_test_detail_{log_ts}.log")
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Main logger: console + file
    _logger = logging.getLogger("step3_test")
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(main_log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    _logger.addHandler(ch)
    _logger.addHandler(fh)

    # Detail logger: file only
    _detail = logging.getLogger("step3_test_detail")
    _detail.setLevel(logging.DEBUG)
    _detail.handlers.clear()

    dfh = logging.FileHandler(detail_log_path, encoding="utf-8")
    dfh.setLevel(logging.DEBUG)
    dfh.setFormatter(fmt)
    _detail.addHandler(dfh)

    return main_log_path, detail_log_path


# ─── 路径 / 数据加载 ─────────────────────────────────────────

def find_latest_merged_model(base_dir: str = "outputs", config_path: str = None) -> str:
    """根据配置生成 run_name，查找对应的 merged_model 或 checkpoint 目录

    优先返回 merged_model，如果不存在则返回最新的 checkpoint
    （用于处理预量化模型无法合并的情况）
    """
    candidates = []

    if config_path and os.path.exists(config_path):
        try:
            cfg = load_config(config_path)
            run_name = generate_run_name(cfg)
            for d in glob.glob(os.path.join(base_dir, f"{run_name}_*", "merged_model")):
                if os.path.isdir(d):
                    candidates.append(d)
        except Exception:
            pass

    if not candidates:
        for d in glob.glob(os.path.join(base_dir, "*", "merged_model")):
            if os.path.isdir(d):
                candidates.append(d)

    if candidates:
        candidates.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        return candidates[0]

    # 回退到查找 checkpoint（预量化模型无法合并的情况）
    if config_path and os.path.exists(config_path):
        try:
            cfg = load_config(config_path)
            run_name = generate_run_name(cfg)
            checkpoints = []
            for run_dir in glob.glob(os.path.join(base_dir, f"{run_name}_*")):
                if os.path.isdir(run_dir):
                    for ckpt in glob.glob(os.path.join(run_dir, "checkpoint-*")):
                        if os.path.isdir(ckpt):
                            checkpoints.append(ckpt)
            if checkpoints:
                checkpoints.sort(key=lambda d: os.path.getmtime(d), reverse=True)
                return checkpoints[0]
        except Exception:
            pass

    # 全局搜索最新的 checkpoint
    checkpoints = []
    for run_dir in glob.glob(os.path.join(base_dir, "*")):
        if os.path.isdir(run_dir):
            for ckpt in glob.glob(os.path.join(run_dir, "checkpoint-*")):
                if os.path.isdir(ckpt):
                    checkpoints.append(ckpt)

    if checkpoints:
        checkpoints.sort(key=lambda d: os.path.getmtime(d), reverse=True)
        return checkpoints[0]

    return ""


def load_system_prompt(data_dir: str = "data") -> str:
    """从训练数据中提取 system prompt"""
    for name in ["mainintent_train.jsonl", "mainintent_val.jsonl"]:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if line:
                item = json.loads(line)
                if item.get("instruction"):
                    return item["instruction"]
    return ""


def load_training_config(model_path: str) -> dict:
    """从模型目录中加载训练配置"""
    run_dir = model_path
    if os.path.basename(os.path.normpath(run_dir)) == "merged_model":
        run_dir = os.path.dirname(run_dir)

    for config_path in [
        os.path.join(run_dir, "config_backup.yaml"),
        os.path.join(run_dir, "llamafactory_train.yaml"),
    ]:
        if os.path.exists(config_path):
            try:
                import yaml
                with open(config_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception:
                continue
    return {}


def extract_hyperparams(config: dict) -> dict:
    """从训练配置中提取关键超参数"""
    if not config:
        return {}
    training = config.get("training", {})
    lora = config.get("lora", {})
    data = config.get("data", {})
    return {
        "finetuning_type": config.get("finetuning_type", training.get("finetuning_type", "lora")),
        "learning_rate": training.get("learning_rate", config.get("learning_rate", "-")),
        "num_epochs": training.get("num_epochs", training.get("num_train_epochs", "-")),
        "batch_size": training.get("per_device_train_batch_size", config.get("per_device_train_batch_size", "-")),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps", config.get("gradient_accumulation_steps", "-")),
        "effective_batch_size": (
            training.get("per_device_train_batch_size", 1) *
            training.get("gradient_accumulation_steps", 1)
        ) if training.get("per_device_train_batch_size") and training.get("gradient_accumulation_steps") else "-",
        "lora_rank": lora.get("rank", config.get("lora_rank", "-")),
        "lora_alpha": lora.get("alpha", config.get("lora_alpha", "-")),
        "lora_dropout": lora.get("dropout", config.get("lora_dropout", "-")),
        "max_seq_length": data.get("max_seq_length", data.get("cutoff_len", "-")),
        "lr_scheduler": training.get("lr_scheduler", training.get("lr_scheduler_type", "-")),
        "warmup_ratio": training.get("warmup_ratio", "-"),
        "weight_decay": training.get("weight_decay", "-"),
    }


def load_test_cases(test_file: str) -> list:
    """加载测试用例，支持 JSON 和 JSONL 格式"""
    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        if content.startswith("{"):
            cases = []
            for line in content.split("\n"):
                line = line.strip()
                if line:
                    item = json.loads(line)
                    cases.append({
                        "input": item.get("input", ""),
                        "expected": item.get("output", ""),
                    })
            return cases
        return json.loads(content)


# ─── 内置默认测试用例 ─────────────────────────────────────────

DEFAULT_TEST_CASES = [
    {"input": "换手机号码",         "expected": "107"},
    {"input": "我要理赔",           "expected": "201"},
    {"input": "附近有服务网点吗",   "expected": "203"},
    {"input": "理赔需要什么材料",   "expected": "204"},
    {"input": "你好",               "expected": "402"},
    {"input": "我想预约绿通",       "expected": "501"},
    {"input": "查保单",             "expected": "502"},
    {"input": "追加保费",           "expected": "503"},
    {"input": "取消绿通",           "expected": "504"},
    {"input": "交费账号变更",       "expected": "532"},
    {"input": "甲状腺结节可以投保吗","expected": "601"},
    {"input": "绿通进度怎么样了",   "expected": "605"},
    {"input": "证件什么时候过期",   "expected": "701"},
    {"input": "万能账户收益多少",   "expected": "702"},
    {"input": "我的VIP等级",        "expected": "901"},
    {"input": "绿通有什么权益",     "expected": "902"},
    {"input": "保单还款",           "expected": "927"},
    {"input": "保单贷款",           "expected": "928"},
    {"input": "是的",               "expected": "301"},
    {"input": "不是",               "expected": "302"},
    {"input": "都不是",             "expected": "303"},
    {"input": "不办了",             "expected": "304"},
    {"input": "帮我看看保单现金价值","expected": "401"},
]


# ─── 辅助函数 ────────────────────────────────────────────────

def show_debug_info(user_input: str, system_prompt: str, enable_thinking: bool, log: logging.Logger):
    """显示调试信息：展示实际发送给模型的请求结构"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_input})

    log.info("")
    log.info(f"{'='*60}")
    log.info(f"  【调试信息】引擎: HuggingFace Transformers | Qwen3 enable_thinking: {enable_thinking}")
    log.info(f"{'='*60}")
    log.info("")
    log.info(f"  messages 结构:")
    log.info(f"  {json.dumps({'messages': messages, 'temperature': 0.1, 'max_new_tokens': 100, 'enable_thinking': enable_thinking}, ensure_ascii=False, indent=2)}")
    log.info("")
    log.info(f"{'='*60}")


def extract_code(response: str, use_mapping: bool = True) -> str:
    """从模型输出中提取意图编码"""
    response = response.strip()
    if not use_mapping:
        return response
    if re.match(r"^\d{3}$", response):
        return response
    m = re.search(r"\b(\d{3})\b", response)
    if m:
        return m.group(1)
    return response


def strip_thinking_block(text: str) -> str:
    """去除 Qwen3 思考链 anuts...=[]块"""
    text = re.sub(r"anuts.*?=[]\s*", "", text, flags=re.DOTALL)
    return text.strip()


# ─── 核心推理 ────────────────────────────────────────────────

def load_model(model_path: str, log: logging.Logger):
    """加载 tokenizer 和模型，支持量化模型 (4bit/8bit)"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    except ImportError:
        log.error("未安装 transformers / torch，请先: pip install transformers torch accelerate")
        sys.exit(1)

    # 检测量化类型
    model_lower = model_path.lower()
    is_unsloth = "unsloth" in model_lower
    is_4bit = "-4bit" in model_lower or "4bit" in model_lower or "bnb-4bit" in model_lower
    is_8bit = "-8bit" in model_lower or "8bit" in model_lower or "bnb-8bit" in model_lower
    quant_bits = 4 if is_4bit else (8 if is_8bit else None)

    log.info(f"[*] 正在加载 tokenizer: {model_path}")

    # 尝试加载 tokenizer - 处理 Unsloth 模型的 BitConfig 问题
    tokenizer = None
    tokenizer_load_path = model_path

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True)
    except (KeyError, Exception) as e:
        log.warning(f"  直接加载 tokenizer 失败: {e}")
        # 尝试从父目录或检查点目录加载
        parent_path = str(Path(model_path).parent)
        success = False

        # 尝试父目录
        try:
            tokenizer = AutoTokenizer.from_pretrained(parent_path, trust_remote_code=True)
            tokenizer_load_path = parent_path
            log.info(f"  从父目录加载 tokenizer 成功")
            success = True
        except:
            pass

        # 尝试 checkpoint 目录
        if not success:
            checkpoints = list(Path(parent_path).glob("checkpoint-*"))
            if checkpoints:
                latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(str(latest_ckpt), trust_remote_code=True)
                    tokenizer_load_path = str(latest_ckpt)
                    log.info(f"  从 checkpoint 加载 tokenizer 成功")
                    success = True
                except:
                    pass

        if not success:
            raise RuntimeError("无法加载 tokenizer，请检查模型路径")

    # 加载模型 - 处理 Unsloth/量化模型的特殊配置
    if is_unsloth or quant_bits:
        quant_info = f"Unsloth {quant_bits}bit" if is_unsloth and quant_bits else (f"{quant_bits}bit" if quant_bits else "Unsloth")
        log.info(f"[*] 检测到 {quant_info} 量化模型，使用特殊加载方式...")

        # 尝试使用量化配置加载
        try:
            from transformers import BitsAndBytesConfig

            # 根据位数创建不同的量化配置
            if quant_bits == 4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif quant_bits == 8:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                # Unsloth 模型默认使用 4bit
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )

            # 尝试从 tokenizer 所在路径加载模型
            model_load_path = tokenizer_load_path if tokenizer_load_path != model_path else model_path

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_load_path,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                log.info(f"[*] {quant_bits if quant_bits else 4}bit 量化模型加载完成")
            except Exception as e2:
                log.warning(f"  从 {model_load_path} 加载失败: {e2}")
                # 最后尝试：使用原始路径但禁用量化（已是反量化的 merged_model）
                log.info(f"  尝试常规加载（反量化模型）...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                )
                log.info(f"[*] 模型加载完成（反量化）")

        except ImportError:
            log.warning(f"  bitsandbytes 未安装，尝试常规加载...")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            log.warning(f"  量化加载失败: {e}")
            # 最后回退：从 checkpoint 加载
            parent_path = Path(model_path).parent
            checkpoints = list(parent_path.glob("checkpoint-*"))
            if checkpoints:
                log.info(f"  尝试从最新的 checkpoint 加载...")
                latest_ckpt = max(checkpoints, key=lambda p: p.stat().st_mtime)
                model = AutoModelForCausalLM.from_pretrained(
                    str(latest_ckpt),
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
                log.info(f"[*] 从 checkpoint 加载完成")
            else:
                raise
    else:
        log.info(f"[*] 正在加载模型（bfloat16 + device_map=auto）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    model.eval()
    log.info(f"[*] 模型加载完成")
    log.info("")
    return tokenizer, model


def run_inference(
    model_path: str,
    test_cases: list,
    system_prompt: str,
    enable_thinking: bool = False,
    debug_mode: bool = False,
    max_new_tokens: int = 100,
    temperature: float = 0.1,
    use_mapping: bool = True,
    log: logging.Logger = None,
    detail_log: logging.Logger = None,
) -> list:
    """使用 HuggingFace Transformers 进行推理"""
    import torch

    if log is None:
        log = logger
    if detail_log is None:
        detail_log = detail_logger

    tokenizer, model = load_model(model_path, log)

    results = []
    total = len(test_cases)

    for i, case in enumerate(test_cases):
        query = case["input"]
        expected = case.get("expected", "")

        if debug_mode and i == 0:
            show_debug_info(query, system_prompt, enable_thinking, log)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        input_len = model_inputs.input_ids.shape[1]

        start_time = time.perf_counter()
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
            )
        latency_ms = (time.perf_counter() - start_time) * 1000

        new_ids = generated_ids[0][input_len:]
        raw_resp = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        response = strip_thinking_block(raw_resp) if enable_thinking else raw_resp
        actual = extract_code(response, use_mapping)
        match = (actual == expected) if expected else None

        results.append({
            "input": query,
            "expected": expected,
            "raw_output": raw_resp,
            "output": response,
            "actual": actual,
            "match": match,
            "latency_ms": round(latency_ms, 2),
        })

        status = "[v]" if match else "[x]" if match is False else "[?]"
        line = (
            f"  [{i+1}/{total}] {status} '{query[:30]}' -> {actual}"
            + (f" (期望: {expected})" if expected and not match else "")
            + (f" [{latency_ms:.0f}ms]" if latency_ms > 0 else "")
        )
        detail_log.info(line)

        # 进度摘要（每 500 条输出一次到主日志）
        if (i + 1) % 500 == 0:
            correct_so_far = sum(1 for r in results if r["match"])
            log.info(f"  进度: {i+1}/{total} ({(i+1)/total*100:.1f}%) | 当前准确率: {correct_so_far}/{i+1} ({correct_so_far/(i+1)*100:.1f}%)")

    return results


# ─── 报告打印 ────────────────────────────────────────────────

def parse_intent_names(system_prompt: str) -> dict:
    """从系统提示词中解析意图编码 → 名称映射"""
    intent_map = {}
    if not system_prompt:
        return intent_map

    # 优先匹配中文全角括号格式：【107】客户信息变更：
    if "【" in system_prompt and "】" in system_prompt:
        pattern0 = r"【(\d{3})】(.+?)(?:[:：]|$)"
        matches = re.findall(pattern0, system_prompt)
        if matches:
            for code, name in matches:
                intent_map[code] = name.strip()
            return intent_map

    # 匹配半角括号格式：[107]客户信息变更：
    if "[" in system_prompt and "]" in system_prompt:
        pattern1 = r"\[(\d{3})\](.+?)[:\n]"
        matches = re.findall(pattern1, system_prompt)
        if matches:
            for code, name in matches:
                intent_map[code] = name.strip()
            return intent_map

    # 匹配短横线格式：客户信息变更-107
    if re.search(r'\S-\d{3}', system_prompt):
        pattern2 = r"([^\[\],\s-]+?)-(\d{3})"
        matches = re.findall(pattern2, system_prompt)
        if matches:
            for name, code in matches:
                intent_map[code] = name.strip()

    return intent_map


def print_report(results: list, system_prompt: str = "", model_path: str = "",
                 training_config: dict = None, engine_name: str = "HuggingFace Transformers",
                 log: logging.Logger = None, detail_log: logging.Logger = None):
    """打印完整测试报告"""
    if log is None:
        log = logger

    total = len(results)
    evaluated = [r for r in results if r["match"] is not None]
    correct = sum(1 for r in evaluated if r["match"])
    intent_names = parse_intent_names(system_prompt) if system_prompt else {}

    model_display = ""
    if model_path:
        try:
            rel_path = os.path.relpath(model_path)
            if rel_path.startswith(".."):
                parts = model_path.replace("\\", "/").split("/")
                model_display = "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            else:
                model_display = rel_path
        except Exception:
            model_display = os.path.basename(model_path)

    # ── 测试报告 ─────────────────────────────
    log.info("")
    log.info(f"{'='*60}")
    if model_display:
        log.info(f"【测试报告 - {model_display}】")
    else:
        log.info(f"【测试报告】")
    log.info(f"{'='*60}")
    log.info("")

    # ── 超参数配置 ─────────────────────────────
    if training_config:
        hyperparams = extract_hyperparams(training_config)
        if hyperparams:
            log.info("【超参数配置】")
            log.info("".join(["-" for _ in range(40)]))
            log.info("  训练方式:")
            log.info(f"    微调类型:     {hyperparams.get('finetuning_type', '-').upper()}")
            log.info(f"    训练轮数:     {hyperparams.get('num_epochs', '-')}")
            log.info("")
            log.info("  优化参数:")
            log.info(f"    学习率:       {hyperparams.get('learning_rate', '-')}")
            log.info(f"    LR Scheduler: {hyperparams.get('lr_scheduler', '-')}")
            log.info(f"    Warmup Ratio: {hyperparams.get('warmup_ratio', '-')}")
            log.info(f"    Weight Decay: {hyperparams.get('weight_decay', '-')}")
            log.info("")
            log.info("  批次配置:")
            log.info(f"    Batch Size:   {hyperparams.get('batch_size', '-')}")
            log.info(f"    梯度累积:     {hyperparams.get('gradient_accumulation_steps', '-')}")
            log.info(f"    有效批次:     {hyperparams.get('effective_batch_size', '-')}")
            log.info("")
            log.info("  LoRA 配置:")
            log.info(f"    Rank:         {hyperparams.get('lora_rank', '-')}")
            log.info(f"    Alpha:        {hyperparams.get('lora_alpha', '-')}")
            log.info(f"    Dropout:      {hyperparams.get('lora_dropout', '-')}")
            log.info("")
            log.info("  其他配置:")
            log.info(f"    Max Seq Len:  {hyperparams.get('max_seq_length', '-')}")
            log.info("".join(["-" for _ in range(40)]))
            log.info("")

    # ── API 调用格式示例 ─────────────────────────────
    if results and system_prompt:
        first_case = results[0]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": first_case["input"]},
        ]
        log.info("【API 调用格式示例】")
        log.info("".join(["-" for _ in range(40)]))
        log.info(f"  引擎: {engine_name}")
        log.info("")
        log.info("  messages = [")
        log.info(f'    {{"role": "system", "content": """{system_prompt}"""}},')
        log.info(f'    {{"role": "user", "content": "{first_case["input"]}"}},')
        log.info("  ]")
        log.info("")
        log.info(f'  模型输出: "{first_case.get("output", first_case.get("actual", ""))}"')
        log.info("".join(["-" for _ in range(40)]))
        log.info("")

    # ── 统计信息 ─────────────────────────────
    log.info(f"  总用例数: {total}")
    log.info(f"  有标注数: {len(evaluated)}")

    if not evaluated:
        log.info(f"{'='*60}")
        return

    accuracy = correct / len(evaluated) * 100
    log.info(f"  总准确率: {accuracy:.2f}% ({correct}/{len(evaluated)})")
    log.info("")

    # ── 性能统计 ─────────────────────────────
    latencies = [r.get("latency_ms", 0) for r in results if r.get("latency_ms")]
    if latencies:
        latencies.sort()
        total_latency = sum(latencies)
        avg_latency = total_latency / len(latencies)
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[int(len(latencies) * 0.95)]
        p99 = latencies[int(len(latencies) * 0.99)]
        total_time_sec = total_latency / 1000
        qps = len(latencies) / total_time_sec if total_time_sec > 0 else 0

        log.info("".join(["━" for _ in range(60)]))
        log.info("【性能统计】")
        log.info("".join(["━" for _ in range(60)]))
        log.info(f"  总耗时:     {total_time_sec:.2f} 秒 ({len(latencies)} 条)")
        log.info(f"  平均耗时:   {avg_latency:.2f} ms")
        log.info(f"  最快:       {latencies[0]:.2f} ms")
        log.info(f"  最慢:       {latencies[-1]:.2f} ms")
        log.info(f"  P50:        {p50:.2f} ms (中位数)")
        log.info(f"  P95:        {p95:.2f} ms (95%% 请求)")
        log.info(f"  P99:        {p99:.2f} ms (99%% 请求)")
        log.info(f"  QPS:        {qps:.2f} 请求/秒")
        log.info("".join(["━" for _ in range(60)]))
        log.info("")

    # 按意图编码统计
    code_stats = defaultdict(lambda: {"correct": 0, "total": 0, "errors": [], "latency_sum": 0})
    for r in evaluated:
        code = r["expected"]
        code_stats[code]["total"] += 1
        if r["match"]:
            code_stats[code]["correct"] += 1
        else:
            code_stats[code]["errors"].append(r)
        if r.get("latency_ms"):
            code_stats[code]["latency_sum"] += r["latency_ms"]

    # ── 各意图分类准确率 ─────────────────────────────
    log.info(f"{'='*60}")
    log.info("【各意图分类准确率】")
    log.info(f"{'='*60}")
    log.info("意图名称                 编码       正确/总数        准确率        平均耗时         状态")
    log.info("".join(["-" for _ in range(76)]))
    for code in sorted(code_stats.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        s = code_stats[code]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        avg_lat = s["latency_sum"] / s["total"] if s["total"] > 0 else 0
        name = intent_names.get(code, "未知意图")[:18]

        if acc == 100:
            flag = "✅"
        elif acc >= 95:
            flag = "🟢 优"
        elif acc >= 90:
            flag = "🟢 良"
        elif acc >= 80:
            flag = "🟡 中"
        elif acc > 0:
            flag = "🔴 差"
        else:
            flag = "❌"

        log.info(f"{name:<24} [{code}]   {s['correct']:>3}/{s['total']:<3}   {acc:>6.2f}%   {avg_lat:>7.2f} ms   {flag}")
    log.info("".join(["-" for _ in range(76)]))
    log.info("")

    # 准确率最低的前10
    log.info("【按准确率排序（最低10个）】")
    sorted_codes = sorted(
        code_stats.items(),
        key=lambda x: x[1]["correct"] / max(x[1]["total"], 1)
    )
    for code, s in sorted_codes[:10]:
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        name = intent_names.get(code, "未知意图")
        log.info(f"  [{code}] {name:<24} {s['correct']}/{s['total']} ({acc:.1f}%)")

    log.info("")
    log.info(f"{'='*60}")

    # ── 混淆矩阵（含 P/R/F1 + Top-10 错误详情）─────────
    print_confusion_matrix(evaluated, intent_names, log, detail_log)


def print_confusion_matrix(evaluated: list, intent_names: dict, log: logging.Logger, detail_log: logging.Logger = None):
    """打印混淆矩阵、P/R/F1 报告（Tab 分隔，可直接复制到 Excel），以及 Top-10 误分详情

    如果提供 detail_log，详细报告（混淆矩阵、P/R/F1、误分详情）将输出到 detail_log
    否则所有内容输出到 log
    """
    # 如果没有提供 detail_log，则所有内容输出到 log
    if detail_log is None:
        detail_log = log

    if not evaluated:
        return

    all_labels = sorted(
        set(r["expected"] for r in evaluated) | set(r["actual"] for r in evaluated),
        key=lambda x: int(x) if x.isdigit() else 0,
    )

    # 构建混淆矩阵 cm[true][pred] = count
    cm: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in evaluated:
        cm[r["expected"]][r["actual"]] += 1

    # per-class TP / FP / FN
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for true_label in all_labels:
        for pred_label in all_labels:
            cnt = cm[true_label][pred_label]
            if true_label == pred_label:
                tp[true_label] += cnt
            else:
                fn[true_label] += cnt
                fp[pred_label] += cnt

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    # ── Section 1: P/R/F1 (Tab 分隔) ─────────────
    detail_log.info("")
    detail_log.info(f"{'='*60}")
    detail_log.info("【分类指标报告 (Precision / Recall / F1)】")
    detail_log.info(f"{'='*60}")
    header_parts = ["意图名称", "编码", "Support", "Precision", "Recall", "F1", "状态"]
    detail_log.info("\t".join(header_parts))

    macro_p, macro_r, macro_f1 = [], [], []
    weighted_p, weighted_r, weighted_f1 = [], [], []
    total_support = len(evaluated)

    for label in all_labels:
        support = tp[label] + fn[label]
        prec = safe_div(tp[label], tp[label] + fp[label])
        rec = safe_div(tp[label], tp[label] + fn[label])
        f1 = safe_div(2 * prec * rec, prec + rec)
        name = intent_names.get(label, "未知意图")

        macro_p.append(prec)
        macro_r.append(rec)
        macro_f1.append(f1)
        if support > 0:
            weighted_p.append(prec * support)
            weighted_r.append(rec * support)
            weighted_f1.append(f1 * support)

        if f1 == 1.0:
            flag = "✅"
        elif f1 >= 0.95:
            flag = "🟢"
        elif f1 >= 0.90:
            flag = "🟢"
        elif f1 >= 0.80:
            flag = "🟡"
        elif f1 > 0:
            flag = "🔴"
        else:
            flag = "❌"

        detail_log.info(f"{name}\t{label}\t{support}\t{prec:.3f}\t{rec:.3f}\t{f1:.3f}\t{flag}")

    n = len(all_labels)
    mp = sum(macro_p) / n if n else 0
    mr = sum(macro_r) / n if n else 0
    mf1 = sum(macro_f1) / n if n else 0
    wp = sum(weighted_p) / total_support if total_support else 0
    wr = sum(weighted_r) / total_support if total_support else 0
    wf1 = sum(weighted_f1) / total_support if total_support else 0

    detail_log.info(f"macro avg\t\t{total_support}\t{mp:.3f}\t{mr:.3f}\t{mf1:.3f}")
    detail_log.info(f"weighted avg\t\t{total_support}\t{wp:.3f}\t{wr:.3f}\t{wf1:.3f}")
    detail_log.info(f"{'='*60}")

    # ── Section 2: 混淆矩阵 (Tab 分隔，仅展示有误分的行) ─────────────
    detail_log.info("")
    detail_log.info("【混淆矩阵（仅展示有误分的行）】")
    detail_log.info("  行 = 真实标签 (True)  |  列 = 预测标签 (Predicted)")
    detail_log.info("  对角线 ✓ = 正确预测   |  非对角线数字 = 误分数量")
    detail_log.info("")

    # 表头: 真实意图 + 编码 + ✓正确 + 各预测编码列 + 总计
    col_labels = all_labels
    header_cells = ["真实意图", "编码", "✓正确"] + [f"[{c}]" for c in col_labels] + ["总计"]
    header = "\t".join(header_cells)
    detail_log.info(header)
    detail_log.info("".join(["-" for _ in range(100)]))

    for true in all_labels:
        name = intent_names.get(true, "未知意图")
        correct_cnt = cm[true][true]
        support = tp[true] + fn[true]
        # 仅展示有误分的行
        if support - correct_cnt == 0:
            continue
        cells = []
        for pred in col_labels:
            cnt = cm[true][pred]
            if pred == true:
                cells.append("✓")
            elif cnt == 0:
                cells.append(".")
            elif cnt >= 10:
                cells.append(f"【{cnt}】")
            else:
                cells.append(str(cnt))
        row = f"{name}\t{true}\t{correct_cnt}\t" + "\t".join(cells) + f"\t{support}"
        detail_log.info(row)

    detail_log.info("".join(["-" for _ in range(100)]))
    detail_log.info("")

    # ── Section 3: Top-10 误分对 + 错误用例详情 ─────────
    confusion_pairs = []
    for true in all_labels:
        for pred, cnt in cm[true].items():
            if pred != true and cnt > 0:
                confusion_pairs.append((cnt, true, pred))
    confusion_pairs.sort(reverse=True)

    if not confusion_pairs:
        detail_log.info("  无错误样本，混淆矩阵为单位矩阵")
        detail_log.info(f"{'='*60}")
        return

    top_n = min(10, len(confusion_pairs))
    top_pairs = confusion_pairs[:top_n]
    top_keys = {(t, p) for _, t, p in top_pairs}

    # 收集 Top 误分对的错误用例
    error_by_pair: dict[tuple, list] = defaultdict(list)
    for r in evaluated:
        if not r["match"] and (r["expected"], r["actual"]) in top_keys:
            error_by_pair[(r["expected"], r["actual"])].append(r)

    detail_log.info(f"{'='*60}")
    detail_log.info(f"【高频误分对 Top-{top_n}】")
    detail_log.info(f"  排名  真实意图                    → 误预测为                    次数  占真实类比例")
    detail_log.info("  " + "".join(["-" for _ in range(70)]))

    for rank, (cnt, true, pred) in enumerate(top_pairs, 1):
        true_name = intent_names.get(true, "未知意图")
        pred_name = intent_names.get(pred, "未知意图")
        support = tp[true] + fn[true]
        ratio = cnt / support * 100 if support > 0 else 0

        detail_log.info(f"  #{rank:<2}  [{true}]{true_name:<20} → [{pred}]{pred_name:<20} {cnt:>3}   ({ratio:>4.1f}%)")

    detail_log.info(f"{'='*60}")
    detail_log.info("")
    detail_log.info("【高频误分对 Top-10 错误用例详情】")
    detail_log.info("")

    for rank, (cnt, true, pred) in enumerate(top_pairs, 1):
        true_name = intent_names.get(true, "未知意图")
        pred_name = intent_names.get(pred, "未知意图")
        support = tp[true] + fn[true]
        ratio = cnt / support * 100 if support > 0 else 0

        separator = "".join(["━" for _ in range(80)])
        detail_log.info(separator)
        detail_log.info(f"#{rank} 真实: [{true}] {true_name}  →  误预测: [{pred}] {pred_name}  ({cnt} 条, {ratio:.1f}%)")
        detail_log.info(separator)
        detail_log.info("")

        pair_errors = error_by_pair.get((true, pred), [])
        for j, err in enumerate(pair_errors, 1):
            detail_log.info(f"  {j}. 输入: \"{err['input']}\"")
            detail_log.info(f"     期望: {err['expected']}  |  预测: {err['actual']}")
            raw = err.get("raw_output", "")
            if raw and raw != err.get("actual", ""):
                detail_log.info(f"     原始输出: {raw[:200]}")
            detail_log.info("")

    detail_log.info(f"{'='*60}")

    return confusion_pairs


# ─── 主入口 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="意图分类模型推理测试（HuggingFace Transformers + Qwen3）"
    )
    parser.add_argument("--model_path", type=str, default=None, help="模型路径（默认根据配置文件自动匹配）")
    parser.add_argument("--config", type=str, default="config/train_config.yaml", help="训练配置文件路径")
    parser.add_argument("--test_file", type=str, default=None, help="测试用例文件 (JSON / JSONL)")
    parser.add_argument("--system_prompt", type=str, default=None, help="系统提示词（默认从训练数据自动提取）")
    parser.add_argument("--output", type=str, default=None, help="推理结果输出路径（默认保存到 run 目录）")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="最大生成 token 数（默认 100）")
    parser.add_argument("--temperature", type=float, default=0.1, help="采样温度（默认 0.1）")
    parser.add_argument("--enable_thinking", action="store_true", help="启用 Qwen3 思考链（默认关闭）")
    parser.add_argument("--debug", action="store_true", default=True, help="调试模式（默认开启）")
    parser.add_argument("--no_debug", action="store_true", help="禁用调试模式")
    parser.add_argument("--raw_output", action="store_true", default=True, help="不提取编码，直接使用模型原始输出（默认开启）")
    parser.add_argument("--extract_code", action="store_true", help="提取编码模式（禁用 raw_output）")
    parser.add_argument("--max_samples", type=int, default=None, help="测试用例数量上限（默认：全部）")
    parser.add_argument("--log_dir", type=str, default=None, help="日志输出目录（默认在模型目录下创建 logs 子目录）")
    args = parser.parse_args()

    if args.no_debug:
        args.debug = False
    if args.extract_code:
        args.raw_output = False

    use_mapping = not args.raw_output

    # ── 确定模型路径 ──────────────────────────────
    model_path = args.model_path
    if not model_path:
        model_path = find_latest_merged_model(config_path=args.config)
        if not model_path:
            print("[!] 未找到合并模型，请指定 --model_path 或先完成训练")
            return
        auto_matched = True
    else:
        auto_matched = False

    # ── 日志 ──────────────────────────────
    if args.log_dir:
        # 用户指定的日志目录
        log_dir = args.log_dir
    else:
        # 默认在模型目录下创建 logs 子目录
        run_dir = model_path
        if os.path.basename(os.path.normpath(run_dir)) == "merged_model":
            run_dir = os.path.dirname(run_dir)
        log_dir = os.path.join(run_dir, "logs")

    global logger, detail_logger
    main_log_path, detail_log_path = setup_logger(log_dir)

    logger.info(f"{'='*60}")
    logger.info(f"  Step 3 — 推理测试 (HuggingFace Transformers)")
    logger.info(f"{'='*60}")
    logger.info(f"  模型路径 : {model_path}")
    logger.info(f"  主日志   : {main_log_path}")
    logger.info(f"  详细日志 : {detail_log_path}")
    if auto_matched:
        logger.info(f"  (自动匹配模型)")
    logger.info(f"  思考链   : {'开启' if args.enable_thinking else '关闭'}")
    logger.info(f"  输出模式 : {'原始标签' if args.raw_output else '提取编码'}")

    # ── 加载测试用例 ──────────────────────────────
    if args.test_file and os.path.exists(args.test_file):
        test_cases = load_test_cases(args.test_file)
        logger.info(f"  测试用例 : {args.test_file} ({len(test_cases)} 条)")
    else:
        val_file = "data/mainintent_val.jsonl"
        if os.path.exists(val_file):
            test_cases = load_test_cases(val_file)
            logger.info(f"  测试用例 : {val_file} ({len(test_cases)} 条，自动选择)")
        else:
            test_cases = DEFAULT_TEST_CASES
            logger.info(f"  测试用例 : 内置默认 ({len(test_cases)} 条)")

    if args.max_samples and args.max_samples < len(test_cases):
        test_cases = test_cases[:args.max_samples]
        logger.info(f"  限制测试 : {args.max_samples} 条")

    # ── 系统提示词 ──────────────────────────────
    system_prompt = args.system_prompt or load_system_prompt()
    if system_prompt:
        logger.info(f"  系统提示词: {len(system_prompt)} 字符")
    else:
        logger.warning(f"  未找到系统提示词，模型可能无法正确分类")

    # ── 推理 ──────────────────────────────
    results = run_inference(
        model_path=model_path,
        test_cases=test_cases,
        system_prompt=system_prompt,
        enable_thinking=args.enable_thinking,
        debug_mode=args.debug,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_mapping=use_mapping,
        log=logger,
        detail_log=detail_logger,
    )

    if not results:
        return

    # ── 报告 ──────────────────────────────
    training_config = load_training_config(model_path)
    print_report(results, system_prompt, model_path, training_config, log=logger)

    # ── 保存 JSON 结果 ──────────────────────────────
    output_path = args.output
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(run_dir, f"inference_results_{timestamp}.json")

    latencies = [r.get("latency_ms", 0) for r in results if r.get("latency_ms")]
    perf_stats = {}
    if latencies:
        latencies_sorted = sorted(latencies)
        perf_stats = {
            "total_latency_ms": round(sum(latencies), 2),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "min_latency_ms": round(latencies_sorted[0], 2),
            "max_latency_ms": round(latencies_sorted[-1], 2),
            "p50_latency_ms": round(latencies_sorted[len(latencies) // 2], 2),
            "p95_latency_ms": round(latencies_sorted[int(len(latencies) * 0.95)], 2),
            "p99_latency_ms": round(latencies_sorted[int(len(latencies) * 0.99)], 2),
            "qps": round(len(latencies) / (sum(latencies) / 1000), 2),
        }

    output_data = {
        "model_path": model_path,
        "engine": "huggingface_transformers",
        "enable_thinking": args.enable_thinking,
        "use_mapping": use_mapping,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "evaluated": len([r for r in results if r["match"] is not None]),
        "correct": sum(1 for r in results if r["match"]),
        "accuracy": round(
            sum(1 for r in results if r["match"]) /
            max(len([r for r in results if r["match"] is not None]), 1) * 100, 2
        ),
        "performance": perf_stats,
        "system_prompt_length": len(system_prompt) if system_prompt else 0,
        "hyperparameters": extract_hyperparams(training_config) if training_config else {},
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    logger.info(f"  结果已保存: {output_path}")


if __name__ == "__main__":
    main()
