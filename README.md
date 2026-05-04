# 保险意图识别微调项目

基于 LLaMA Factory 的 LoRA 微调项目，用于保险业务意图分类识别。

## 项目特点

- **一键执行**：数据准备 → 训练 → 测试 → HTTP API 部署
- **批量训练**：支持多个模型/提示词组合自动训练对比
- **多种测试方式**：本地推理 + HTTP API 测试
- **多种部署框架**：vLLM / SGLang / Ollama

---

## 快速开始

### 1. 安装依赖

```bash
pip install llamafactory openpyxl pyyaml tensorboard
```

### 2. 一键执行完整流程

```bash
# 使用默认配置执行完整流程
python run_pipeline.py

# 指定提示词 ID
python run_pipeline.py --prompt_id PROMPT_001

# 指定 Excel 文件
python run_pipeline.py --input 蒸馏模型-数据汇总-260421.xlsx
```

### 3. 查看结果

训练结果保存在 `outputs/<run_name>/` 目录下，包含：
- 合并后的模型：`merged_model/`
- 训练日志：`logs/`
- 推理结果：`inference_results_*.json`

---

## 核心脚本说明

### 📊 step1_prepare.py — 数据准备

从 Excel 文件生成 Alpaca JSONL 格式的微调数据。

**用法：**
```bash
# 自动选择最新 Excel 文件
python step1_prepare.py

# 指定 Excel 文件
python step1_prepare.py --input 蒸馏模型-数据汇总-260421.xlsx

# 指定提示词 ID
python step1_prepare.py --prompt_id PROMPT_004

# 使用原始标签（不使用编码映射）
python step1_prepare.py --raw_output

# 指定输出目录
python step1_prepare.py --output_dir data
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 自动选最新 | Excel 文件路径 |
| `--output_dir` | `data` | 输出目录 |
| `--dataset_name` | `mainintent` | 数据集名称 |
| `--prompt_id` | 配置文件/无 | 提示词 ID（优先级：命令行 > 配置文件 > 默认逻辑） |
| `--raw_output` | `false` | 不使用编码映射，直接使用人工标注结果作为 output |

**自动完成：**
- 从"提示词" sheet 提取系统提示词
- 从"业务编码映射" sheet 构建意图编码映射
- 生成 `mainintent_train.jsonl` 和 `mainintent_val.jsonl`
- 注册数据集到 `data/dataset_info.json`

---

### 🚀 step2_train.py — 模型训练

基于 LLaMA Factory 的自动化训练脚本。

**用法：**
```bash
# 使用默认配置训练
python step2_train.py

# 指定配置文件
python step2_train.py --config config/train_config.yaml

# 指定运行名称
python step2_train.py --run_name exp01_lora_r8

# 仅验证配置，不实际训练
python step2_train.py --dry_run

# 跳过 LoRA 权重合并
python step2_train.py --skip_merge

# 跳过模型评估
python step2_train.py --skip_eval
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `config/train_config.yaml` | 训练配置文件路径 |
| `--run_name` | 自动生成 | 运行名称（影响输出目录名） |
| `--skip_merge` | `false` | 跳过 LoRA 权重合并 |
| `--skip_eval` | `false` | 跳过模型评估 |
| `--dry_run` | `false` | 仅验证配置和环境，不实际训练 |

**自动完成：**
- 环境检查（Python、PyTorch、CUDA、GPU、磁盘空间）
- 模型完整性检查，缺失时自动下载
- 数据格式验证
- 启动训练并实时输出日志
- 选择最佳 checkpoint
- 合并 LoRA 权重为完整模型
- 生成训练报告

---

### 🧪 step3_test.py — 本地推理测试

使用 HuggingFace Transformers 加载合并后的模型进行推理测试。

**用法：**
```bash
# 自动选择最新模型，使用验证集测试
python step3_test.py

# 指定模型路径
python step3_test.py --model_path outputs/run_20260423_111922/merged_model

# 指定测试文件
python step3_test.py --test_file data/mainintent_val.jsonl

# 限制测试用例数量（快速验证）
python step3_test.py --max_samples 50

# 调整生成参数
python step3_test.py --temperature 0.3 --max_new_tokens 200

# 提取编码模式（从输出中正则提取 3 位意图编码）
python step3_test.py --extract_code
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 自动选最新 | 合并模型路径 |
| `--test_file` | 自动选验证集 | 测试用例文件 |
| `--system_prompt` | 从训练数据提取 | 系统提示词 |
| `--output` | 自动生成 | 结果输出路径 |
| `--max_new_tokens` | `100` | 最大生成 token 数 |
| `--temperature` | `0.1` | 采样温度 |
| `--enable_thinking` | `false` | 启用 Qwen3 思考链 |
| `--debug` | `true` | 调试模式：打印第一条请求的完整结构 |
| `--raw_output` | `true` | 不提取编码，直接使用模型原始输出 |
| `--extract_code` | `false` | 提取编码模式（正则提取 3 位数字） |
| `--max_samples` | 全部 | 测试用例数量上限 |

---

### 🌐 step4_test_http.py — HTTP API 测试

模拟外部 OpenAI API 调用，测试模型的意图分类准确率。

**用法：**
```bash
# 自动部署 + 测试（默认 vLLM）
python step4_test_http.py

# 跳过部署（服务已运行）
python step4_test_http.py --skip_serve

# 限制测试条数
python step4_test_http.py --max_samples 50

# 使用并发测试
python step4_test_http.py --workers 4
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `config/train_config.yaml` | 训练配置文件路径 |
| `--serve_config` | `config/serve_config.yaml` | 部署配置文件路径 |
| `--model_path` | 自动选最新 | 合并模型路径 |
| `--test_file` | 自动选验证集 | 测试用例文件 |
| `--skip_serve` | `false` | 跳过启动服务（假设服务已运行） |
| `--max_samples` | 全部 | 测试用例数量上限 |
| `--workers` | `1` | 并发线程数 |

**支持的部署框架：**
- **vLLM**（默认）：生产首选，吞吐量最高
- **SGLang**：结构化生成强，性能优秀
- **Ollama**：本地部署最简单

---

### 🔧 serve.py — 模型服务管理

独立的模型服务启停脚本，仅负责服务生命周期管理。

**用法：**
```bash
# 启动服务（自动匹配最新模型）
python serve.py start

# 指定模型路径
python serve.py start --model_path outputs/run_*/merged_model

# 停止服务
python serve.py stop

# 查看服务状态
python serve.py status

# 重启服务
python serve.py restart

# 启动后不等待就绪
python serve.py start --no_wait
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `action` | 必填 | `start` / `stop` / `status` / `restart` |
| `--config` | `config/train_config.yaml` | 训练配置文件路径 |
| `--serve_config` | `config/serve_config.yaml` | 部署配置文件路径 |
| `--model_path` | 自动选最新 | 合并模型路径 |
| `--timeout` | `300` | 等待服务就绪的超时秒数 |
| `--no_wait` | `false` | 启动后不等待服务就绪 |

---

### 🖥️ train_webui.py — WebUI 可视化训练

启动 LLaMA Factory WebUI，通过浏览器进行交互式训练。

**用法：**
```bash
# 启动 WebUI（默认端口 7860）
python train_webui.py

# 指定配置文件（预填参数到 WebUI）
python train_webui.py --config config/train_config.yaml

# 创建公开链接（远程访问）
python train_webui.py --share

# 跳过模型检查（加速启动）
python train_webui.py --skip_model_check
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `config/train_config.yaml` | 配置文件路径 |
| `--run_name` | 自动生成 | 运行名称 |
| `--skip_model_check` | `false` | 跳过模型检查 |
| `--share` | `false` | 创建 Gradio 公开链接 |

---

### 🔄 run_pipeline.py — 一键流水线

串联四个阶段：数据准备 → 模型训练 → 本地测试 → HTTP API 测试。

**用法：**
```bash
# 完整流程
python run_pipeline.py

# 指定提示词
python run_pipeline.py --prompt_id PROMPT_001

# 跳过某个阶段
python run_pipeline.py --skip_prepare
python run_pipeline.py --skip_train
python run_pipeline.py --skip_test
python run_pipeline.py --skip_http

# 仅执行某个阶段
python run_pipeline.py --only_prepare
python run_pipeline.py --only_train
python run_pipeline.py --only_test
python run_pipeline.py --only_http

# 测试时限制用例数量
python run_pipeline.py --max_test_samples 100
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 自动选最新 | Excel 数据文件路径 |
| `--prompt_id` | 无 | 提示词 ID |
| `--config` | `config/train_config.yaml` | 训练配置文件路径 |
| `--skip_prepare` | `false` | 跳过数据准备 |
| `--skip_train` | `false` | 跳过模型训练 |
| `--skip_test` | `false` | 跳过本地推理测试 |
| `--skip_http` | `false` | 跳过 HTTP API 测试 |
| `--skip_serve` | `false` | 跳过模型服务部署 |
| `--only_prepare` | `false` | 仅执行数据准备 |
| `--only_train` | `false` | 仅执行模型训练 |
| `--only_test` | `false` | 仅执行本地推理测试 |
| `--only_http` | `false` | 仅执行 HTTP API 测试 |
| `--max_test_samples` | 全部 | 测试用例数量上限 |
| `--workers` | `1` | HTTP 测试并发线程数 |

---

### 📦 run_pipeline_batch.py — 批量训练

批量执行多个训练任务，自动对比不同模型/提示词组合的效果。

**用法：**
```bash
# 使用默认任务列表
python run_pipeline_batch.py

# 自定义任务列表
python run_pipeline_batch.py --tasks "Qwen/Qwen3-4B:001,Qwen/Qwen3-4B:004,Qwen/Qwen3-8B:001"

# 设置间隔时间（分钟）
python run_pipeline_batch.py --interval 10

# 不等待，连续执行
python run_pipeline_batch.py --no-wait

# 只显示将要执行的任务，不实际运行
python run_pipeline_batch.py --dry-run
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `config/train_config.yaml` | 训练配置文件路径 |
| `--tasks` | 默认任务列表 | 任务列表，格式: '模型1:prompt1,模型2:prompt2' |
| `--interval` | `5` | 任务之间的间隔时间（分钟） |
| `--dry-run` | `false` | 只显示将要执行的任务，不实际运行 |
| `--no-wait` | `false` | 不等待，连续执行 |

**量化配置自动检测：**
- 模型名包含 `-4bit` 或 `bnb-4bit` → QLoRA 4bit
- 模型名包含 `-8bit` 或 `bnb-8bit` → QLoRA 8bit
- 其他模型 → LoRA（非量化）

---

## 配置文件说明

### config/train_config.yaml — 训练配置

```yaml
# 模型配置
model:
  name_or_path: "unsloth/Qwen3-4B-unsloth-bnb-4bit"  # 模型路径
  trust_remote_code: true

# 数据配置
data:
  dataset_name: "mainintent"     # 数据集名称
  dataset_dir: "data"            # 数据集目录
  template: "qwen3_nothink"      # 模板类型
  max_seq_length: 4096           # 最大序列长度
  prompt_id: null                # 提示词 ID

# 训练配置
training:
  stage: "sft"                   # 训练阶段
  finetuning_type: "lora"        # 微调类型
  num_epochs: 2                  # 训练轮次
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5.0e-5
  bf16: true

# LoRA 配置
lora:
  rank: 8
  alpha: 16
  dropout: 0.05
  target_modules: "all"

# 量化配置
quantization:
  enable: false
  bits: 4
```

### config/serve_config.yaml — 部署配置

```yaml
# 服务端配置
serving:
  framework: "vllm"              # vllm / sglang / ollama
  host: "127.0.0.1"
  port: 8000
  served_model_name: "qwen3-intent"
  gpu_memory_utilization: 0.9
  max_model_len: 2048

# 客户端配置
client:
  mode: "openai"                 # openai / requests
  timeout: 30
  max_retries: 3
  workers: 1

# 测试配置
test:
  temperature: 0.1
  max_new_tokens: 100
  enable_thinking: false
  extract_code: true
  log_request_body: true
```

---

## 常见问题

### Q: 显存不足怎么办？

在 `config/train_config.yaml` 中调整：
```yaml
training:
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8

data:
  max_seq_length: 1024

quantization:
  enable: true
  bits: 4
```

### Q: 模型下载慢？

脚本会自动使用 ModelScope（国内速度快）。手动下载：
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-8B')"
```

### Q: 如何断点续训？

将配置中的模型路径指向 checkpoint 子目录：
```yaml
model:
  name_or_path: "outputs/run_20260423_111922/checkpoint-500"
```

### Q: 预量化模型（Unsloth）无法合并？

这是正常现象，预量化模型的 LoRA 权重无法合并。脚本会自动检测并跳过合并步骤，直接使用 checkpoint 进行推理。

---

## 项目结构

```
├── step1_prepare.py            # 数据准备
├── step2_train.py              # 模型训练
├── step3_test.py               # 本地推理测试
├── step4_test_http.py          # HTTP API 测试
├── serve.py                    # 模型服务管理
├── train_webui.py              # WebUI 可视化训练
├── run_pipeline.py             # 一键流水线
├── run_pipeline_batch.py       # 批量训练
├── config/
│   ├── train_config.yaml       # 训练配置
│   └── serve_config.yaml       # 部署配置
├── data/                       # 数据集目录
│   ├── dataset_info.json
│   ├── mainintent_train.jsonl
│   └── mainintent_val.jsonl
└── outputs/                    # 训练输出
    └── <run_name>/
        ├── logs/
        ├── merged_model/
        └── inference_results_*.json
```

---

## 模板速查

| 模型系列 | template 值 |
|---------|-------------|
| Qwen3（分类/简单任务） | `qwen3_nothink` |
| Qwen3（通用） | `qwen3` |
| Qwen 1/2/2.5 | `qwen` |
| Llama-3.x | `llama3` |
| ChatGLM3 | `chatglm3` |
