# LLaMA Factory 单卡微调全流程指南

> 阿里云 PAI 平台 | 单卡训练 | 意图识别微调

## 📁 项目结构

```
├── prepare_data.py             # 数据准备脚本（Excel -> JSONL）
├── train.py                    # 主训练脚本（核心入口）
├── inference_test.py           # 训练后推理验证
├── 蒸馏模型-数据汇总-YYMMDD.xlsx  # 原始数据文件
├── config/
│   └── train_config.yaml       # 训练配置文件
├── data/                       # 数据集目录（脚本自动生成）
│   ├── dataset_info.json       # 数据集注册表
│   ├── intent_cls.jsonl        # 训练集（LLaMA Factory 自动切分 80/20 用于训练和验证）
│   └── intent_cls_val.jsonl    # 测试集（训练后评估用，不参与训练）
└── outputs/                    # 所有运行记录（运行后自动创建）
    ├── prepare_data.log        # 数据准备日志（所有运行追加）
    └── run_20240101_120000/    # 训练运行目录
        ├── llamafactory_train.yaml
        ├── config_backup.yaml
        ├── logs/
        │   ├── run_20240101_120000.log      # 训练日志
        │   └── inference_20240102_100000.log # 推理日志
        ├── run_report.json
        ├── run_summary.txt
        ├── inference_results_20240102_100000.json
        └── merged_model/
```

---

## 🚀 完整微调流程

### 第一步：安装依赖

```bash
pip install llamafactory
pip install openpyxl
pip install flash-attn --no-build-isolation  # 可选，仅 fa2 模式需要（默认 sdpa 无需安装）
pip install tensorboard
```

---

### 第二步：准备数据

#### 2.1 数据来源

原始数据为 Excel 文件（`蒸馏模型-数据汇总-YYMMDD.xlsx`），包含以下 sheet：

| Sheet | 说明 |
|-------|------|
| 提示词 | System 提示词模板（是否训练=是 的行生效） |
| 业务编码映射 | 意图名称与编码的映射关系 |
| 训练集 | 客户问题 + 人工标注结果 |
| 验证集 | 客户问题 + 人工标注结果（训练后评估用，不参与训练） |

#### 2.2 运行数据准备脚本

```bash
# 自动选择日期最新的 Excel 文件
python prepare_data.py

# 指定 Excel 文件
python prepare_data.py --input 蒸馏模型-数据汇总-260421.xlsx

# 指定数据集名称
python prepare_data.py --dataset_name intent_cls
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | 自动选最新 | Excel 文件路径，无参数时按文件名日期选最新 |
| `--output_dir` | `data` | 输出目录 |
| `--dataset_name` | `intent_cls` | 数据集名称 |

**日志输出：** 运行日志同时输出到控制台和 `outputs/prepare_data.log`（追加模式）。

**脚本会自动：**
- 从"提示词"sheet 提取系统提示词
- 从"业务编码映射"sheet 构建意图编码映射（以提示词为准）
- 将"训练集"sheet 转换为 `intent_cls.jsonl`（训练数据，LLaMA Factory 自动 80/20 切分）
- 将"验证集"sheet 转换为 `intent_cls_val.jsonl`（训练后评估用，不参与训练）
- 注册训练数据集到 `data/dataset_info.json`
- 更新 `config/train_config.yaml` 中的 `dataset_name`

**数据流说明：**
```
Excel "训练集" → intent_cls.jsonl ─→ LLaMA Factory (val_size=0.2 自动切分)
                                      ├── 80% 实际训练
                                      └── 20% 训练中验证（eval_steps 评估）

Excel "验证集" → intent_cls_val.jsonl → inference_test.py 训练后评估
```

**输出示例：**
```
  [训练集]
    [客户信息变更-107] : 772 条
    [理赔报案-201] : 172 条
    [查询服务网点-203] : 300 条
    ...
    合计: 5731 条

  [测试集（训练后评估用）]
    [保单贷款-928] : 216 条
    ...
    合计: 3776 条
```

#### 2.3 生成的数据格式

每行一条 JSON，Alpaca 格式：

```json
{
  "instruction": "你是一个专业的保险业务意图识别助手...", 
  "input": "换手机号码", 
  "output": "107"}
```

| 字段 | 内容 |
|------|------|
| `instruction` | 系统提示词（完整的意图分类规则） |
| `input` | 客户问题 |
| `output` | 意图编码（如 107、201 等） |

---

### 第三步：确认训练配置

数据准备完成后，`config/train_config.yaml` 会自动更新 `dataset_name`。

如需修改模型路径或其他参数，编辑 `config/train_config.yaml`：

```yaml
model:
  name_or_path: "/path/to/your/model"   # ← 改成你的模型路径

data:
  dataset_name: "intent_cls"             # ← 已自动更新
  template: "llama3"                     # ← 根据模型选择模板

training:
  num_epochs: 3
  per_device_train_batch_size: 2
  learning_rate: 1.0e-4
  val_size: 0.2                          # ← 训练数据中 20% 用于验证集

lora:
  rank: 8
```

**显存参考表（LoRA 单卡）**

| 显存    | 推荐 batch_size | 推荐 max_seq_length |
|---------|-----------------|----------------------|
| 16 GB   | 1               | 1024                 |
| 24 GB   | 2               | 2048                 |
| 40 GB   | 4               | 4096                 |
| 80 GB   | 8               | 8192                 |

---

### 第四步：启动训练

```bash
# 单卡训练
python train.py --config config/train_config.yaml

# 指定运行名称（推荐，便于管理）
python train.py --config config/train_config.yaml --run_name exp01_lora_r8

# 验证配置（不实际训练）
python train.py --config config/train_config.yaml --dry_run

# 训练完跳过合并
python train.py --config config/train_config.yaml --skip_merge
```

---

### 第五步：监控训练过程

```bash
# 查看实时日志
tail -f outputs/run_xxx/logs/run_xxx.log

# 启动 TensorBoard 查看 loss 曲线
tensorboard --logdir outputs/ --port 6006
# 浏览器打开 http://localhost:6006
```

---

### 第六步：验证训练结果

```bash
# 自动选择最新的合并模型（推荐）
python inference_test.py

# 手动指定模型路径
python inference_test.py --model_path outputs/run_xxx/merged_model

# 使用验证集做批量测试
python inference_test.py --test_file data/intent_cls_val.jsonl

# 指定输出路径
python inference_test.py --output outputs/run_xxx/inference_results.json
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 自动选最新 | 合并模型路径，无参数时自动选 `outputs/run_*/merged_model` 中最新的 |
| `--test_file` | 内置用例 | 测试用例文件，支持 JSON 和 JSONL 格式 |
| `--template` | `llama3` | 对话模板 |
| `--output` | 自动生成 | 结果输出路径，默认保存在模型目录下 |

**脚本会自动：**
- 从训练数据中提取系统提示词（意图分类规则）
- 将用户问题 + 系统提示词发送给模型推理
- 从模型输出中提取意图编码，与期望结果对比
- 输出总准确率、各意图编码准确率、错误用例详情
- 运行日志保存至 `outputs/run_xxx/logs/inference_{timestamp}.log`
- 测试结果保存至 `outputs/run_xxx/inference_results_{timestamp}.json`

**输出示例：**
```
  [v] '换手机号码' -> 107
  [x] '你好' -> 401 (期望: 402)
  ...
============================================================
测试报告
============================================================
  总用例数: 23
  有标注数: 23
  准确率:   87.0% (20/23)

  [各意图准确率]
    [v] [107] 1/1 (100%)
    [v] [201] 1/1 (100%)
    [x] [402] 0/1 (0%)
    ...

  [错误用例] (3 条)
    '你好' -> 预测: 401, 正确: 402
============================================================
```

---

### 第七步：查看训练报告

训练结束后，自动生成两个报告文件：

**`run_report.json`** - 完整机器可读报告（含所有参数）

**`run_summary.txt`** - 人类可读摘要，示例：
```
============================================================
训练运行摘要: exp01_lora_r8
============================================================
状态:     ✓ 成功
时间:     2024-01-01T14:30:00
耗时:     02h 15m 33s
模型:     meta-llama/Meta-Llama-3.1-8B-Instruct
训练方式: lora
数据集:   intent_cls (5731 条)
Epochs:  3
LR:      0.0001
============================================================
```

---

## 🔧 常见问题

**Q: OOM（显存不足）怎么办？**
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
max_seq_length: 1024
quantization:
  enable: true
  bits: 4
```

**Q: 训练 loss 不下降？**
- 检查数据格式是否正确（运行 `--dry_run` 排查）
- 尝试降低学习率（`1e-4` → `5e-5`）
- 检查 `template` 是否与模型匹配

**Q: 如何断点续训？**
```yaml
# 将模型路径指向具体的 checkpoint 子目录
model:
  name_or_path: "outputs/run_20240101_120000/checkpoint-500"
# 或者使用 LLaMA Factory 的 resume_from_checkpoint 参数
```

**Q: LoRA vs QLoRA 怎么选？**
- 显存充足（≥ 24GB）→ LoRA（更快，效果略好）
- 显存不足（< 16GB）→ QLoRA（量化节省显存）

---

## 📊 模型对应模板速查

| 模型系列            | template 值     |
|--------------------|-----------------|
| Llama-3.x          | llama3          |
| Qwen 1/2/2.5       | qwen            |
| Mistral/Mixtral    | mistral         |
| ChatGLM3           | chatglm3        |
| Baichuan2          | baichuan2       |
| InternLM2          | intern2         |
| Gemma              | gemma           |
| DeepSeek-V2        | deepseekcoder   |
| Yi                 | yi              |
