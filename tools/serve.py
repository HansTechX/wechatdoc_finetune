#!/usr/bin/env python3
"""
模型服务管理脚本 - 支持启动、停止、状态查看和测试
支持 vLLM / SGLang / Ollama 三种部署框架

用法:
  python tools/serve.py start                        # 启动服务（自动匹配模型）
  python tools/serve.py start --model_path /path/to/model
  python tools/serve.py stop                         # 停止服务
  python tools/serve.py status                       # 查看服务状态
  python tools/serve.py restart                      # 重启服务
"""

import os
import sys
import json
import time
import signal
import argparse
import subprocess
import tempfile
import urllib.request
import urllib.error
import random

# 获取项目根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from step3_test import find_latest_merged_model

# PID 文件路径（用于 stop / status 操作）
PID_FILE = "/tmp/model_serve.pid"
META_FILE = "/tmp/model_serve.meta.json"  # 保存 base_url、framework 等元信息


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


def load_val_samples(val_path: str, count: int = 5) -> list:
    """从验证集加载样本，用于生成测试用例"""
    if not os.path.exists(val_path):
        return []

    samples = []
    with open(val_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                samples.append(data)
                if len(samples) >= count:
                    break

    return samples


# ─── 启动命令构造 ──────────────────────────────────────────────

def build_vllm_cmd(model_path: str, cfg: dict) -> list:
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


def build_ollama_modelfile(model_path: str, model_name: str) -> str:
    """写入临时 Modelfile，返回路径"""
    modelfile_path = os.path.join(tempfile.gettempdir(), f"Modelfile_{model_name}")
    with open(modelfile_path, "w") as f:
        f.write(f"FROM {model_path}\n")
    return modelfile_path


# ─── 服务生命周期 ──────────────────────────────────────────────

def start_server(model_path: str, cfg: dict, log_dir: str = "logs") -> tuple:
    """
    启动模型服务，写入 PID 文件和元信息文件。
    返回 (pid, base_url)，失败返回 (None, "")。
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
    print(f"  模型路径  : {model_path}")
    print(f"  服务端日志: {server_log_path}")

    if framework == "ollama":
        model_name = cfg.get("serving", {}).get("served_model_name", "qwen3-intent")
        modelfile_path = build_ollama_modelfile(model_path, model_name)

        create_cmd = ["ollama", "create", model_name, "-f", modelfile_path]
        print(f"  创建 Ollama 模型: {' '.join(create_cmd)}")
        result = subprocess.run(create_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [!] Ollama create 失败:\n{result.stderr}")
            server_log_file.close()
            return None, ""

        print(f"  模型创建成功")
        serve_cmd = ["ollama", "serve"]
        print(f"  启动服务: {' '.join(serve_cmd)}")
        proc = subprocess.Popen(
            serve_cmd,
            stdout=server_log_file,
            stderr=subprocess.STDOUT,
        )
        base_url = f"http://{host}:11434"

    elif framework in ("vllm", "sglang"):
        cmd = build_vllm_cmd(model_path, cfg) if framework == "vllm" else build_sglang_cmd(model_path, cfg)
        print(f"  命令: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=server_log_file,
            stderr=subprocess.STDOUT,
        )
        base_url = f"http://{host}:{port}"

    else:
        print(f"  [!] 不支持的框架: {framework}")
        server_log_file.close()
        return None, ""

    # 写入 PID 文件
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))

    # 写入元信息
    meta = {
        "pid": proc.pid,
        "base_url": base_url,
        "framework": framework,
        "model_path": model_path,
        "log_path": server_log_path,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  PID       : {proc.pid}  (已写入 {PID_FILE})")
    print(f"  服务地址  : {base_url}")
    return proc.pid, base_url


def wait_for_server(base_url: str, timeout: int = 300, framework: str = "vllm") -> bool:
    """轮询等待服务就绪"""
    health_url = f"{base_url}/api/tags" if framework == "ollama" else f"{base_url}/v1/models"
    print(f"\n  等待服务就绪: {health_url}")
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(urllib.request.Request(health_url), timeout=5) as resp:
                if resp.status == 200:
                    elapsed = time.time() - start
                    print(f"  ✓ 服务就绪 (耗时 {elapsed:.1f}s)")
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(3)

    print(f"\n  [!] 服务启动超时 ({timeout}s)")
    return False


def stop_server() -> bool:
    """读取 PID 文件并停止服务"""
    if not os.path.exists(PID_FILE):
        print("[!] 未找到 PID 文件，服务可能未运行或已手动停止")
        return False

    with open(PID_FILE) as f:
        pid_str = f.read().strip()

    if not pid_str.isdigit():
        print(f"[!] PID 文件内容无效: {pid_str}")
        return False

    pid = int(pid_str)
    print(f"\n{'='*60}")
    print(f"【停止模型服务】PID: {pid}")
    print(f"{'='*60}")

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"  已发送 SIGTERM -> PID {pid}")

        # 等待进程退出（最多 15s）
        for _ in range(15):
            time.sleep(1)
            try:
                os.kill(pid, 0)  # 进程存在则不抛异常
                sys.stdout.write(".")
                sys.stdout.flush()
            except ProcessLookupError:
                print(f"\n  ✓ 进程已退出")
                break
        else:
            # 超时强杀
            print(f"\n  [!] 超时，发送 SIGKILL -> PID {pid}")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    except ProcessLookupError:
        print(f"  [!] PID {pid} 不存在，服务可能已退出")

    # 清理文件
    for f in (PID_FILE, META_FILE):
        if os.path.exists(f):
            os.remove(f)

    print(f"  PID 文件已清理")
    return True


def show_status() -> None:
    """显示当前服务状态"""
    print(f"\n{'='*60}")
    print(f"【服务状态】")
    print(f"{'='*60}")

    if not os.path.exists(META_FILE):
        print("  状态: 未运行（无元信息文件）")
        return

    with open(META_FILE, encoding="utf-8") as f:
        meta = json.load(f)

    pid = meta.get("pid")
    alive = False
    try:
        os.kill(pid, 0)
        alive = True
    except (ProcessLookupError, TypeError):
        pass

    status_str = "✓ 运行中" if alive else "✗ 已停止"
    print(f"  状态      : {status_str}")
    print(f"  PID       : {pid}")
    print(f"  框架      : {meta.get('framework')}")
    print(f"  服务地址  : {meta.get('base_url')}")
    print(f"  模型路径  : {meta.get('model_path')}")
    print(f"  启动时间  : {meta.get('start_time')}")
    print(f"  日志文件  : {meta.get('log_path')}")

    if alive:
        # 探测健康端点
        base_url = meta.get("base_url", "")
        framework = meta.get("framework", "vllm")
        health_url = f"{base_url}/api/tags" if framework == "ollama" else f"{base_url}/v1/models"
        try:
            with urllib.request.urlopen(urllib.request.Request(health_url), timeout=5) as resp:
                ready = resp.status == 200
        except Exception:
            ready = False
        print(f"  API 就绪  : {'✓ 是' if ready else '✗ 否（进程存在但 API 未响应）'}")


def print_curl_examples(base_url: str, model_name: str, val_path: str, count: int = 3) -> None:
    """从验证集读取样本并打印 curl 测试用例"""
    samples = load_val_samples(val_path, count)

    print(f"\n{'='*60}")
    print(f"【curl 实例】可直接复制执行测试（来自验证集）")
    print(f"{'='*60}")
    print("")

    if not samples:
        print(f"  [!] 未找到验证集文件: {val_path}")
        print(f"\n通用测试用例:")
        # 构建通用测试用例的请求体
        request_body = {
            "model": model_name,
            "messages": [{"role": "user", "content": "测试"}],
            "temperature": 0.1,
            "max_tokens": 100
        }
        body_str = json.dumps(request_body, ensure_ascii=False, indent=4)
        print(f"""curl -X POST "{base_url}/v1/chat/completions" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer test" \\
  -d '{body_str}'""")
        return

    for i, sample in enumerate(samples, 1):
        user_input = sample.get("input", "")
        expected_output = sample.get("output", "")

        # 构造用户消息
        instruction = sample.get("instruction", "")
        messages_json = []
        if instruction:
            # 截断过长的 instruction
            sys_content = instruction[:500] + "..." if len(instruction) > 500 else instruction
            messages_json.append({"role": "system", "content": sys_content})
        messages_json.append({"role": "user", "content": user_input})

        # 构建完整的请求体
        request_body = {
            "model": model_name,
            "messages": messages_json,
            "temperature": 0.1,
            "max_tokens": 50
        }
        body_str = json.dumps(request_body, ensure_ascii=False, indent=4)

        print(f"# 实例 {i}: {user_input[:40]}{'...' if len(user_input) > 40 else ''}")
        if expected_output:
            print(f"# 期望输出: {expected_output}")
        print(f'curl -X POST "{base_url}/v1/chat/completions" \\')
        print('  -H "Content-Type: application/json" \\')
        print('  -H "Authorization: Bearer test" \\')
        print(f"  -d '{body_str}'")

        if i < count:
            print("")


# ─── 主入口 ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="模型服务管理 - 启动 / 停止 / 状态查看",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "action",
        choices=["start", "stop", "status", "restart"],
        help=(
            "start   : 启动模型服务\n"
            "stop    : 停止模型服务\n"
            "status  : 查看服务状态\n"
            "restart : 重启模型服务"
        ),
    )
    parser.add_argument(
        "--config", type=str, default="config/train_config.yaml",
        help="训练配置文件路径（用于自动匹配模型目录和数据集）",
    )
    parser.add_argument(
        "--serve_config", type=str, default="config/serve_config.yaml",
        help="部署配置文件路径",
    )
    parser.add_argument(
        "--model_path", type=str, default=None,
        help="模型路径（不指定则自动从训练配置匹配最新合并模型）",
    )
    parser.add_argument(
        "--timeout", type=int, default=300,
        help="等待服务就绪的超时秒数（默认 300）",
    )
    parser.add_argument(
        "--log_dir", type=str, default=None,
        help="服务端日志目录（默认: <model_path>/logs）",
    )
    parser.add_argument(
        "--no_wait", action="store_true",
        help="启动后不等待服务就绪，直接返回",
    )
    parser.add_argument(
        "--val_samples", type=int, default=3,
        help="显示的验证集测试用例数量（默认 3）",
    )

    args = parser.parse_args()

    # 切换到项目根目录
    os.chdir(PROJECT_ROOT)

    # ── stop / status 不需要加载模型配置 ──
    if args.action == "stop":
        stop_server()
        return

    if args.action == "status":
        show_status()
        return

    if args.action == "restart":
        stop_server()
        time.sleep(2)
        # 继续执行 start 逻辑（下方）

    # ── start / restart ──
    serve_cfg = load_serve_config(args.serve_config)
    framework = serve_cfg.get("serving", {}).get("framework", "vllm")

    model_path = args.model_path
    if not model_path:
        model_path = find_latest_merged_model(config_path=args.config)
        if not model_path:
            print("[!] 未找到合并模型，请通过 --model_path 显式指定")
            sys.exit(1)
        print(f"(自动匹配模型: {model_path})")

    # 确定日志目录
    log_dir = args.log_dir
    if not log_dir:
        run_dir = model_path
        if os.path.basename(os.path.normpath(run_dir)) == "merged_model":
            run_dir = os.path.dirname(run_dir)
        log_dir = os.path.join(run_dir, "logs")

    pid, base_url = start_server(model_path, serve_cfg, log_dir=log_dir)
    if pid is None:
        print("[!] 服务启动失败")
        sys.exit(1)

    if args.no_wait:
        print("\n[--no_wait] 跳过就绪等待，服务在后台启动中")
        print(f"可使用以下命令查看状态:\n  python tools/serve.py status")
        return

    if not wait_for_server(base_url, timeout=args.timeout, framework=framework):
        print("[!] 服务就绪超时，请检查日志:", log_dir)
        sys.exit(1)

    # 获取验证集路径（从 train_config.yaml 读取数据集配置）
    val_path = None
    try:
        import yaml
        with open(args.config, "r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f)
            dataset_name = train_cfg.get("data", {}).get("dataset_name", "mainintent")
            dataset_dir = train_cfg.get("data", {}).get("dataset_dir", "data")
            val_path = os.path.join(dataset_dir, f"{dataset_name}_val.jsonl")
    except Exception as e:
        print(f"[!] 无法读取数据集配置: {e}")
        val_path = "data/mainintent_val.jsonl"  # 默认路径

    # 打印测试用例
    model_name = serve_cfg.get("serving", {}).get("served_model_name", "qwen3-intent")
    print_curl_examples(base_url, model_name, val_path, args.val_samples)

    # 打印其他提示
    print(f"\n{'='*60}")
    print(f"停止服务:\n  python tools/serve.py stop")
    print(f"查看状态:\n  python tools/serve.py status")


if __name__ == "__main__":
    main()
