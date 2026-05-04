#!/usr/bin/env python3
"""
HTTP API 测试脚本 - 模拟外部 OpenAI API 调用，测试合并模型的意图分类准确率
支持 vLLM / SGLang / Ollama 三种部署框架，OpenAI SDK / requests 两种客户端模式
"""

import os
import sys
import re
import json
import time
import signal
import logging
import argparse
import subprocess
import tempfile
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from step2_train import generate_run_name, load_config
from step3_test import (
    find_latest_merged_model,
    load_system_prompt,
    load_training_config,
    extract_hyperparams,
    load_test_cases,
    DEFAULT_TEST_CASES,
    extract_code,
    strip_thinking_block,
    parse_intent_names,
    print_report,
)


# ─── 日志配置 ────────────────────────────────────────────────
logger = logging.getLogger("train_webui")


def setup_logger(log_dir: str) -> str:
    log_ts = datetime.now().strftime("%m%d_%H%M")
    log_path = os.path.join(log_dir, f"train_webui_{log_ts}.log")
    os.makedirs(log_dir, exist_ok=True)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    _logger = logging.getLogger("train_webui")
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    _logger.addHandler(ch)
    _logger.addHandler(fh)
    return log_path


# ─── 配置加载 ──────────────────────────────────────────────────

def load_serve_config(path: str) -> dict:
    """加载部署配置"""
    try:
        import yaml
    except ImportError:
        print("[!] 需要 pyyaml: pip install pyyaml")
        sys.exit(1)
    if not os.path.exists(path):
        print(f"[!] 配置文件不存在: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ─── 服务启动命令构造 ──────────────────────────────────────────

def build_vllm_cmd(model_path: str, cfg: dict) -> list:
    """构造 vLLM 启动命令"""
    srv = cfg.get("serving", {})
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--host", srv.get("host", "127.0.0.1"),
        "--port", str(srv.get("port", 8000)),
        "--served-model-name", srv.get("served_model_name", "qwen3-intent"),
        "--trust-remote-code",
        "--dtype", "bfloat16",
        "--gpu-memory-utilization", str(srv.get("gpu_memory_utilization", 0.9)),
        "--max-model-len", str(srv.get("max_model_len", 2048)),
    ]
    extra = srv.get("extra_args", "")
    if extra:
        cmd.extend(extra.split())
    return cmd


def build_sglang_cmd(model_path: str, cfg: dict) -> list:
    """构造 SGLang 启动命令"""
    srv = cfg.get("serving", {})
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--host", srv.get("host", "127.0.0.1"),
        "--port", str(srv.get("port", 8000)),
        "--mem-fraction-static", str(srv.get("gpu_memory_utilization", 0.9)),
        "--trust-remote-code",
    ]
    extra = srv.get("extra_args", "")
    if extra:
        cmd.extend(extra.split())
    return cmd


def build_ollama_cmd(model_path: str, cfg: dict) -> tuple:
    """
    构造 Ollama 启动命令。
    返回 (启动命令列表, 环境变量dict, 清理函数)。
    Ollama 需要先 create 模型再 serve，比较特殊。
    """
    srv = cfg.get("serving", {})
    model_name = srv.get("served_model_name", "qwen3-intent")

    # 生成临时 Modelfile
    modelfile_content = f'FROM {model_path}\n'
    modelfile_path = os.path.join(tempfile.gettempdir(), f"Modelfile_{model_name}")
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    # ollama create
    create_cmd = ["ollama", "create", model_name, "-f", modelfile_path]
    # ollama serve (会启动 API server，默认端口 11434)
    serve_cmd = ["ollama", "serve"]

    def cleanup():
        if os.path.exists(modelfile_path):
            os.remove(modelfile_path)

    return create_cmd, serve_cmd, cleanup


# ─── 服务生命周期管理 ──────────────────────────────────────────

def start_server(model_path: str, cfg: dict, log_dir: str = "logs") -> tuple:
    """
    启动模型服务，返回 (process, base_url, cleanup_fn)。
    process 为 None 表示启动失败或跳过。
    服务端日志写入 log_dir/server.log，避免 PIPE 缓冲区满导致服务阻塞。
    """
    framework = cfg.get("serving", {}).get("framework", "vllm")
    host = cfg.get("serving", {}).get("host", "127.0.0.1")
    port = cfg.get("serving", {}).get("port", 8000)

    os.makedirs(log_dir, exist_ok=True)
    server_log_path = os.path.join(log_dir, "server.log")
    server_log_file = open(server_log_path, "w", encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"【启动模型服务】框架: {framework}")
    print(f"{'='*60}")
    print(f"  服务端日志: {server_log_path}")

    if framework == "ollama":
        create_cmd, serve_cmd, cleanup = build_ollama_cmd(model_path, cfg)
        # 先 create
        print(f"  创建 Ollama 模型: {' '.join(create_cmd)}")
        result = subprocess.run(create_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [!] Ollama create 失败: {result.stderr}")
            server_log_file.close()
            return None, "", cleanup
        print(f"  模型创建成功")

        # 启动 serve
        print(f"  启动服务: {' '.join(serve_cmd)}")
        proc = subprocess.Popen(
            serve_cmd,
            stdout=server_log_file,
            stderr=subprocess.STDOUT,
        )
        base_url = f"http://{host}:11434"
    else:
        if framework == "vllm":
            cmd = build_vllm_cmd(model_path, cfg)
        elif framework == "sglang":
            cmd = build_sglang_cmd(model_path, cfg)
        else:
            print(f"  [!] 不支持的框架: {framework}")
            server_log_file.close()
            return None, "", None

        print(f"  命令: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=server_log_file,
            stderr=subprocess.STDOUT,
        )
        base_url = f"http://{host}:{port}"

    return proc, base_url, None


def wait_for_server(base_url: str, timeout: int = 300, framework: str = "vllm") -> bool:
    """轮询等待服务就绪"""
    import urllib.request
    import urllib.error

    # Ollama 的 health 端点不同
    if framework == "ollama":
        health_url = f"{base_url}/api/tags"
    else:
        health_url = f"{base_url}/v1/models"

    print(f"  等待服务就绪: {health_url}")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(health_url)
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    elapsed = time.time() - start
                    print(f"  服务就绪 (耗时 {elapsed:.1f}s)")
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(3)

    print(f"  [!] 服务启动超时 ({timeout}s)")
    return False


def stop_server(proc, framework: str = "vllm"):
    """关闭服务进程"""
    if proc is None:
        return

    print(f"\n  关闭服务进程 (PID: {proc.pid})...")
    try:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    except Exception as e:
        print(f"  关闭服务异常: {e}")


# ─── HTTP 请求（OpenAI SDK） ──────────────────────────────────

def send_openai_request(client, base_url: str, model_name: str, messages: list,
                        temperature: float, max_tokens: int, timeout: int,
                        log_request: bool, log_response: bool) -> tuple:
    """
    用 OpenAI SDK 发送请求。
    返回 (response_text, latency_ms, raw_response_dict)
    """
    request_body = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if log_request:
        print(f"\n  [请求] POST {base_url}/v1/chat/completions")
        # 简化 system 消息显示
        display_body = json.loads(json.dumps(request_body))
        for msg in display_body.get("messages", []):
            if msg["role"] == "system" and len(msg["content"]) > 200:
                msg["content"] = msg["content"][:200] + "..."
        print(json.dumps(display_body, ensure_ascii=False, indent=2))

    start_time = time.perf_counter()
    response = client.chat.completions.create(**request_body)
    latency_ms = (time.perf_counter() - start_time) * 1000

    response_text = response.choices[0].message.content.strip()

    raw_response = {
        "id": response.id,
        "model": response.model,
        "choices": [
            {
                "index": c.index,
                "message": {"role": c.message.role, "content": c.message.content},
                "finish_reason": c.finish_reason,
            }
            for c in response.choices
        ],
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        } if response.usage else None,
    }

    if log_response:
        status = "200 OK" if response_text else "Empty"
        print(f"  [响应] {status} ({latency_ms:.0f}ms)")
        print(json.dumps(raw_response, ensure_ascii=False, indent=2))

    return response_text, latency_ms, raw_response


# ─── HTTP 请求（requests） ────────────────────────────────────

def send_requests_request(session, base_url: str, model_name: str, messages: list,
                          temperature: float, max_tokens: int, timeout: int,
                          log_request: bool, log_response: bool) -> tuple:
    """
    用 requests Session 发送 HTTP 请求（复用 TCP 连接）。
    返回 (response_text, latency_ms, raw_response_dict)
    """
    url = f"{base_url}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    request_body = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if log_request:
        print(f"\n  [请求] POST {url}")
        display_body = json.loads(json.dumps(request_body))
        for msg in display_body.get("messages", []):
            if msg["role"] == "system" and len(msg["content"]) > 200:
                msg["content"] = msg["content"][:200] + "..."
        print(json.dumps(display_body, ensure_ascii=False, indent=2))

    start_time = time.perf_counter()
    resp = session.post(url, json=request_body, headers=headers, timeout=timeout)
    latency_ms = (time.perf_counter() - start_time) * 1000

    if resp.status_code != 200:
        error_text = resp.text[:500]
        if log_response:
            print(f"  [响应] {resp.status_code} ({latency_ms:.0f}ms)")
            print(f"  {error_text}")
        return f"HTTP {resp.status_code}: {error_text}", latency_ms, {"error": error_text, "status_code": resp.status_code}

    raw_response = resp.json()
    response_text = raw_response["choices"][0]["message"]["content"].strip()

    if log_response:
        print(f"  [响应] {resp.status_code} OK ({latency_ms:.0f}ms)")
        print(json.dumps(raw_response, ensure_ascii=False, indent=2))

    return response_text, latency_ms, raw_response


# ─── HTTP 推理主循环 ──────────────────────────────────────────

_thread_local = threading.local()


def _get_thread_client(client_mode: str, base_url: str, timeout: int):
    """获取或创建线程本地 HTTP 客户端（并发模式每个线程独立）"""
    if not hasattr(_thread_local, "client"):
        if client_mode == "openai":
            import httpx
            from openai import OpenAI
            _thread_local.client = OpenAI(
                base_url=f"{base_url}/v1",
                api_key="test",
                timeout=httpx.Timeout(timeout, connect=10.0),
            )
        else:
            import requests
            _thread_local.client = requests.Session()
    return _thread_local.client


def run_http_inference(
    base_url: str,
    test_cases: list,
    system_prompt: str,
    cfg: dict,
    debug_mode: bool = True,
    workers: int = 1,
) -> list:
    """通过 HTTP API 推理，返回结果列表。workers > 1 时使用线程池并发。"""
    framework = cfg.get("serving", {}).get("framework", "vllm")
    model_name = cfg.get("serving", {}).get("served_model_name", "qwen3-intent")
    client_mode = cfg.get("client", {}).get("mode", "openai")
    timeout = cfg.get("client", {}).get("timeout", 30)
    max_retries = cfg.get("client", {}).get("max_retries", 3)

    test_cfg = cfg.get("test", {})
    max_new_tokens = test_cfg.get("max_new_tokens", 100)
    temperature = test_cfg.get("temperature", 0.1)
    enable_thinking = test_cfg.get("enable_thinking", False)
    extract_code_flag = test_cfg.get("extract_code", True)
    log_request = test_cfg.get("log_request_body", True)

    use_mapping = extract_code_flag
    total = len(test_cases)
    send_fn = send_openai_request if client_mode == "openai" else send_requests_request

    # ── 调试信息 ──
    if debug_mode:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": test_cases[0]["input"]})
        print(f"\n{'='*80}")
        print(f"【调试信息】引擎: HTTP API ({framework}) | 客户端: {client_mode} | 并发: {workers}")
        print(f"{'='*80}")
        print(f"\n  base_url: {base_url}/v1")
        print(f"  model: {model_name}")
        print(f"  messages 结构:")
        for i, msg in enumerate(messages):
            content = msg["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"    [{i+1}] {msg['role']}: {content}")
        print(f"  temperature: {temperature} | max_tokens: {max_new_tokens}")
        print(f"  enable_thinking: {enable_thinking}")
        print(f"{'='*80}\n")

    # ── 单条请求处理（顺序/并发共用） ──
    def _process_one(index: int, case: dict, verbose: bool = False) -> dict:
        query = case["input"]
        expected = case.get("expected", "")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        client = _get_thread_client(client_mode, base_url, timeout) if workers > 1 else http_client

        response_text = ""
        latency_ms = 0
        for attempt in range(max_retries):
            try:
                response_text, latency_ms, _ = send_fn(
                    client, base_url, model_name, messages,
                    temperature, max_new_tokens, timeout,
                    verbose, verbose,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    response_text = f"ERROR: {e}"

        if enable_thinking:
            response_text = strip_thinking_block(response_text)

        actual = extract_code(response_text, use_mapping)
        match = (actual == expected) if expected else None

        return {
            "input": query,
            "expected": expected,
            "raw_output": response_text,
            "output": response_text,
            "actual": actual,
            "match": match,
            "latency_ms": round(latency_ms, 2),
        }

    # ── 创建共享客户端（顺序模式） ──
    if client_mode == "openai":
        import httpx
        from openai import OpenAI
        http_client = OpenAI(
            base_url=f"{base_url}/v1",
            api_key="test",
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
    else:
        import requests
        http_client = requests.Session()

    try:
        if workers <= 1:
            # ── 顺序模式（保持原有行为） ──
            VERBOSE_LIMIT = 3
            results = []
            for i, case in enumerate(test_cases):
                if i > 0 and i % 200 == 0:
                    if hasattr(http_client, "close"):
                        http_client.close()
                    if client_mode == "openai":
                        http_client = OpenAI(
                            base_url=f"{base_url}/v1",
                            api_key="test",
                            timeout=httpx.Timeout(timeout, connect=10.0),
                        )
                    else:
                        http_client = requests.Session()
                    print(f"  [i] 刷新 HTTP 客户端 (第 {i+1} 条)")

                verbose = (i < VERBOSE_LIMIT) and log_request
                r = _process_one(i, case, verbose)
                results.append(r)

                status = "[v]" if r["match"] else "[x]" if r["match"] is False else "[?]"
                print(
                    f"  [{i+1}/{total}] {status} '{r['input'][:30]}' -> {r['actual']}"
                    + (f" (期望: {r['expected']})" if r['expected'] and not r['match'] else "")
                    + (f" [{r['latency_ms']:.0f}ms]" if r['latency_ms'] > 0 else "")
                )
            return results

        # ── 并发模式 ──
        print(f"  并发模式: {workers} 线程 | 总用例: {total}")

        results = [None] * total
        completed = [0]
        count_lock = threading.Lock()
        progress_step = max(1, total // 20)

        def _job(index, case):
            r = _process_one(index, case, verbose=False)
            with count_lock:
                completed[0] += 1
                done = completed[0]
            if done % progress_step == 0 or done == total:
                print(f"  进度: {done}/{total} ({done/total*100:.1f}%)")
            return index, r

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_job, i, case)
                for i, case in enumerate(test_cases)
            ]
            for future in as_completed(futures):
                idx, r = future.result()
                results[idx] = r

        # 按顺序打印全部结果
        for i, r in enumerate(results):
            status = "[v]" if r["match"] else "[x]" if r["match"] is False else "[?]"
            print(
                f"  [{i+1}/{total}] {status} '{r['input'][:30]}' -> {r['actual']}"
                + (f" (期望: {r['expected']})" if r['expected'] and not r['match'] else "")
                + (f" [{r['latency_ms']:.0f}ms]" if r['latency_ms'] > 0 else "")
            )

        return results

    finally:
        if hasattr(http_client, "close"):
            http_client.close()


# ─── 主入口 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="HTTP API 测试 - 模拟外部 OpenAI API 调用测试模型准确率"
    )
    parser.add_argument(
        "--config", type=str, default="config/train_config.yaml",
        help="训练配置文件路径（用于自动匹配模型目录）"
    )
    parser.add_argument(
        "--serve_config", type=str, default="config/serve_config.yaml",
        help="部署配置文件路径"
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="合并模型路径（默认自动匹配）"
    )
    parser.add_argument(
        "--test_file", type=str, default=None,
        help="测试用例文件 (JSON / JSONL)"
    )
    parser.add_argument(
        "--system_prompt", type=str, default=None,
        help="系统提示词（默认从训练数据自动提取）"
    )
    parser.add_argument(
        "--skip_serve", action="store_true",
        help="跳过启动服务（假设服务已运行）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="推理结果输出路径（默认保存到 run 目录）"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="测试用例数量上限"
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="并发线程数（默认 1，顺序执行）"
    )
    args = parser.parse_args()

    # 加载配置
    train_cfg = load_config(args.config) if os.path.exists(args.config) else {}
    serve_cfg = load_serve_config(args.serve_config)

    framework = serve_cfg.get("serving", {}).get("framework", "vllm")
    client_mode = serve_cfg.get("client", {}).get("mode", "openai")
    test_cfg = serve_cfg.get("test", {})
    enable_thinking = test_cfg.get("enable_thinking", False)
    extract_code_flag = test_cfg.get("extract_code", True)
    max_samples = args.max_samples or test_cfg.get("max_samples")
    use_mapping = extract_code_flag
    workers = args.workers if args.workers > 1 else serve_cfg.get("client", {}).get("workers", 1)

    # 确定模型路径
    model_path = args.model_path
    if not model_path:
        model_path = find_latest_merged_model(config_path=args.config)
        if not model_path:
            print("[!] 未找到合并模型，请指定 --model_path 或先完成训练")
            return
        print(f"(自动匹配模型: {model_path})")

    # 加载测试用例
    if args.test_file and os.path.exists(args.test_file):
        test_cases = load_test_cases(args.test_file)
        print(f"测试用例: {args.test_file} ({len(test_cases)} 条)")
    else:
        val_file = "data/mainintent_val.jsonl"
        if os.path.exists(val_file):
            test_cases = load_test_cases(val_file)
            print(f"测试用例: {val_file} ({len(test_cases)} 条，自动选择)")
        else:
            test_cases = DEFAULT_TEST_CASES
            print(f"测试用例: 内置默认 ({len(test_cases)} 条)")

    if max_samples and max_samples < len(test_cases):
        test_cases = test_cases[:max_samples]
        print(f"限制测试: {max_samples} 条")

    # 系统提示词
    system_prompt = args.system_prompt or load_system_prompt()
    if system_prompt:
        print(f"系统提示词: {len(system_prompt)} 字符")
    else:
        print("[!] 未找到系统提示词")

    # 打印启动信息
    host = serve_cfg.get("serving", {}).get("host", "127.0.0.1")
    port = serve_cfg.get("serving", {}).get("port", 8000)
    print(f"模型路径: {model_path}")
    print(f"部署框架: {framework} | 客户端: {client_mode}")
    print(f"服务地址: http://{host}:{port}")
    print(f"输出模式: {'提取编码' if use_mapping else '原始标签'}")

    # 日志输出到 run 目录
    run_dir = model_path
    if os.path.basename(os.path.normpath(run_dir)) == "merged_model":
        run_dir = os.path.dirname(run_dir)
    log_dir = os.path.join(run_dir, "logs")

    global logger
    log_path = setup_logger(log_dir)

    base_url = f"http://{host}:{port}"
    proc = None

    try:
        # 启动服务
        if not args.skip_serve:
            proc, base_url, cleanup = start_server(model_path, serve_cfg, log_dir=log_dir)
            if proc is None and not args.skip_serve:
                print("[!] 服务启动失败")
                return
            if not wait_for_server(base_url, framework=framework):
                print("[!] 服务就绪检查超时")
                stop_server(proc, framework)
                return
        else:
            print(f"\n[跳过] 服务启动（使用已运行的服务: {base_url}）")

        # 推理
        wall_start = time.perf_counter()
        results = run_http_inference(
            base_url=base_url,
            test_cases=test_cases,
            system_prompt=system_prompt,
            cfg=serve_cfg,
            debug_mode=True,
            workers=workers,
        )
        wall_time_ms = (time.perf_counter() - wall_start) * 1000

        if not results:
            return

        # 加载训练配置
        training_config = load_training_config(model_path)

        # 报告
        engine_name = f"HTTP API ({framework})"
        print_report(results, system_prompt, model_path, training_config, engine_name=engine_name, log=logger)

        # 保存 JSON 结果
        output_path = args.output
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(run_dir, f"inference_http_results_{timestamp}.json")

        # 性能统计
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
                "wall_time_ms": round(wall_time_ms, 2),
                "qps": round(len(latencies) / (wall_time_ms / 1000), 2),
            }

        if workers > 1:
            print(f"\n  并发总耗时: {wall_time_ms/1000:.2f}s ({workers} 线程, QPS: {perf_stats.get('qps', 0):.1f})")

        output_data = {
            "model_path": model_path,
            "engine": "http_api",
            "serving_framework": framework,
            "client_mode": client_mode,
            "workers": workers,
            "base_url": base_url,
            "enable_thinking": enable_thinking,
            "use_mapping": use_mapping,
            "temperature": test_cfg.get("temperature", 0.1),
            "max_new_tokens": test_cfg.get("max_new_tokens", 100),
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
        print(f"\n结果已保存: {output_path}")

    except KeyboardInterrupt:
        print(f"\n[!] 用户中断")

    finally:
        if proc and not args.skip_serve:
            stop_server(proc, framework)


if __name__ == "__main__":
    main()
