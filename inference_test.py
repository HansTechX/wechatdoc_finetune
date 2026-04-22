#!/usr/bin/env python3
"""
模型推理测试脚本 - 意图分类专用
训练完成后，用于验证模型的意图识别准确率
"""

import os
import sys
import re
import glob
import json
import argparse
from collections import defaultdict
from datetime import datetime


class _Tee:
    """同时输出到控制台和日志文件"""
    def __init__(self, console, file):
        self.console = console
        self.file = file
    def write(self, text):
        self.console.write(text)
        self.file.write(text)
    def flush(self):
        self.console.flush()
        self.file.flush()


def find_latest_merged_model(base_dir: str = "outputs") -> str:
    """在 outputs/ 下找日期最新的 merged_model 目录
    路径格式: outputs/run_{YYYYMMDD_HHMMSS}/merged_model
    """
    pattern = os.path.join(base_dir, "run_*", "merged_model")
    candidates = []
    for d in glob.glob(pattern):
        if os.path.isdir(d):
            # 提取 run_ 后面的时间戳
            basename = os.path.basename(os.path.dirname(d))
            m = re.search(r"run_(\d{8}_\d{6})", basename)
            if m:
                candidates.append((m.group(1), d))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def load_system_prompt(data_dir: str = "data") -> str:
    """从训练数据中提取 system prompt"""
    for name in ["intent_cls.jsonl", "intent_cls_val.jsonl"]:
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


def load_test_cases(test_file: str) -> list:
    """加载测试用例，支持 JSON 和 JSONL 格式"""
    with open(test_file, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if not content:
            return []
        # JSONL: 每行一个 JSON
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
        # JSON array
        return json.loads(content)


DEFAULT_TEST_CASES = [
    {"input": "换手机号码", "expected": "107"},
    {"input": "我要理赔", "expected": "201"},
    {"input": "附近有服务网点吗", "expected": "203"},
    {"input": "理赔需要什么材料", "expected": "204"},
    {"input": "你好", "expected": "402"},
    {"input": "我想预约绿通", "expected": "501"},
    {"input": "查保单", "expected": "502"},
    {"input": "追加保费", "expected": "503"},
    {"input": "取消绿通", "expected": "504"},
    {"input": "交费账号变更", "expected": "532"},
    {"input": "甲状腺结节可以投保吗", "expected": "601"},
    {"input": "绿通进度怎么样了", "expected": "605"},
    {"input": "证件什么时候过期", "expected": "701"},
    {"input": "万能账户收益多少", "expected": "702"},
    {"input": "我的VIP等级", "expected": "901"},
    {"input": "绿通有什么权益", "expected": "902"},
    {"input": "保单还款", "expected": "927"},
    {"input": "保单贷款", "expected": "928"},
    {"input": "是的", "expected": "301"},
    {"input": "不是", "expected": "302"},
    {"input": "都不是", "expected": "303"},
    {"input": "不办了", "expected": "304"},
    {"input": "帮我看看保单现金价值", "expected": "401"},
]


def extract_code(response: str) -> str:
    """从模型输出中提取意图编码"""
    response = response.strip()
    # 直接就是编码
    if re.match(r"^\d{3}$", response):
        return response
    # 提取第一个3位数字
    m = re.search(r"\b(\d{3})\b", response)
    if m:
        return m.group(1)
    return response


def run_inference(model_path: str, test_cases: list, system_prompt: str, template: str) -> list:
    """使用 LLaMA Factory 进行推理"""
    try:
        from llamafactory.chat import ChatModel
    except ImportError:
        print("[!] 未安装 LLaMA Factory，请先 pip install llamafactory")
        return []

    args = {
        "model_name_or_path": model_path,
        "template": template,
        "finetuning_type": "lora" if os.path.exists(os.path.join(model_path, "adapter_config.json")) else "full",
        "infer_dtype": "auto",
    }

    model = ChatModel(args)
    results = []

    for i, case in enumerate(test_cases):
        query = case["input"]
        expected = case.get("expected", "")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        response = ""
        for token in model.stream_chat(messages):
            response += token

        actual = extract_code(response)
        match = (actual == expected) if expected else None

        results.append({
            "input": query,
            "expected": expected,
            "raw_output": response.strip(),
            "actual": actual,
            "match": match,
        })

        status = "[v]" if match else "[x]" if match is False else "[?]"
        print(f"  [{i+1}/{len(test_cases)}] {status} '{query[:30]}' -> {actual}" +
              (f" (期望: {expected})" if expected and not match else ""))

    return results


def print_report(results: list):
    """打印测试报告"""
    total = len(results)
    evaluated = [r for r in results if r["match"] is not None]
    correct = sum(1 for r in evaluated if r["match"])

    print(f"\n{'='*60}")
    print(f"测试报告")
    print(f"{'='*60}")
    print(f"  总用例数: {total}")
    print(f"  有标注数: {len(evaluated)}")

    if evaluated:
        accuracy = correct / len(evaluated) * 100
        print(f"  准确率:   {accuracy:.1f}% ({correct}/{len(evaluated)})")

        # 按意图编码统计
        code_stats = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in evaluated:
            code = r["expected"]
            code_stats[code]["total"] += 1
            if r["match"]:
                code_stats[code]["correct"] += 1

        print(f"\n  [各意图准确率]")
        for code in sorted(code_stats.keys(), key=lambda x: int(x) if x.isdigit() else 0):
            s = code_stats[code]
            acc = s["correct"] / s["total"] * 100
            mark = "[v]" if acc == 100 else "[x]" if acc == 0 else "[~]"
            print(f"    {mark} [{code}] {s['correct']}/{s['total']} ({acc:.0f}%)")

    # 错误详情
    errors = [r for r in evaluated if not r["match"]]
    if errors:
        print(f"\n  [错误用例] ({len(errors)} 条)")
        for r in errors:
            print(f"    '{r['input'][:40]}' -> 预测: {r['actual']}, 正确: {r['expected']}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="意图分类模型推理测试")
    parser.add_argument("--model_path", type=str, default=None, help="模型路径（默认自动选最新的合并模型）")
    parser.add_argument("--test_file", type=str, default=None, help="测试用例文件 (JSON/JSONL)")
    parser.add_argument("--template", type=str, default="llama3", help="对话模板")
    parser.add_argument("--system_prompt", type=str, default=None, help="系统提示词（默认从训练数据提取）")
    parser.add_argument("--output", type=str, default=None, help="结果输出路径")
    args = parser.parse_args()

    # 确定模型路径
    model_path = args.model_path
    if not model_path:
        model_path = find_latest_merged_model()
        if not model_path:
            print("[!] 未找到合并模型，请指定 --model_path 或先完成训练")
            return
        print(f"(自动选择最新模型)")

    # 加载测试用例
    if args.test_file and os.path.exists(args.test_file):
        test_cases = load_test_cases(args.test_file)
        print(f"测试用例: {args.test_file} ({len(test_cases)} 条)")
    else:
        test_cases = DEFAULT_TEST_CASES
        print(f"测试用例: 内置默认 ({len(test_cases)} 条)")

    # 系统提示词
    system_prompt = args.system_prompt or load_system_prompt()
    if system_prompt:
        print(f"系统提示词: {len(system_prompt)} 字符")
    else:
        print("[!] 未找到系统提示词，模型可能无法正确分类")

    print(f"模型路径: {model_path}")
    print(f"对话模板: {args.template}")

    # 日志输出到 run 目录下的 logs/
    run_dir = model_path
    if os.path.basename(os.path.normpath(run_dir)) == "merged_model":
        run_dir = os.path.dirname(run_dir)
    log_dir = os.path.join(run_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_name = f"inference_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _log_file = open(os.path.join(log_dir, log_name), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, _log_file)

    # 推理
    results = run_inference(model_path, test_cases, system_prompt, args.template)

    if not results:
        return

    # 报告
    print_report(results)

    # 保存结果
    output_path = args.output
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(run_dir, f"inference_results_{timestamp}.json")

    output_data = {
        "model_path": model_path,
        "template": args.template,
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "evaluated": len([r for r in results if r["match"] is not None]),
        "correct": sum(1 for r in results if r["match"]),
        "results": results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {output_path}")

    sys.stdout = _log_file.console
    _log_file.close()


if __name__ == "__main__":
    main()
