# Decoupled Speculation Benchmark

## 1. 目录介绍

`benchmark/decoupled_spec/` 提供一套面向 decoupled speculation 的可配置、可观测、可追溯 benchmark。一次实验由四个独立组件协作完成：

| 组件 | 入口 | 主要职责 |
| --- | --- | --- |
| Verifier server | `server-side/verifier_server.py` | 启动 target model HTTP server，接收 client batch，执行 draft verification，并向 drafter 返回 verify 结果 |
| Drafter server | `server-side/drafter_server.py` | 启动普通 decode draft model HTTP server，接收 verify 结果并生成 K-step、F=1 的线性 draft chain |
| Client | `client-side/client.py` | 解析 dataset，加载 target tokenizer，应用 chat template，在本地 tokenize，并通过一次 streaming `/generate` 请求提交整个 batch |
| Observability collector | `common/collector.py` | 周期性采集 verifier 和 drafter 的服务状态，保存原始时间序列 |
| Plot scripts | `plot/*.py` | 分别从稳定的 client/observability 产物生成 latency、speculative、overview 图和单轮报告 |

四个组件共享同一个 `RUN_DIR`。配置、输入样本、streaming 事件、请求级指标、服务状态、图表、Git 信息和校验和都汇总到该目录中，因此一轮实验可以被完整检查和复现。

### 1.1 运行时关系

```text
                         decoupled-spec data plane
                    draft tokens ─────────────────►
┌──────────────────┐                               ┌──────────────────┐       batch POST /generate       ┌────────────────────────┐
│  drafter server  │                               │ verifier server  │ ◄────────────────────────────── │ client                 │
│  draft model     │ ◄──────────────────────────── │ target model     │ ──────────────────────────────► │ 解析 dataset           │
└──────────────────┘          verify results       └──────────────────┘       streaming SSE + index      │ apply chat template    │
          ▲                                               ▲                                             │ target tokenize        │
          │ GET /v1/loads                                 │ GET /v1/loads                               └────────────────────────┘
          └──────────────── observability collector ──────┘
```

正式请求沿以下路径运行：

1. Client 从 dataset 选取 `batch.size` 条样本，用 target tokenizer 完成 chat template 渲染和 tokenization。
2. Client 将全部 `input_ids` 放入一次 HTTP `POST /generate`，发送给 verifier。
3. Verifier 与 drafter 通过 decoupled-spec data plane 循环交换 draft tokens 和 verify results。
4. Verifier 通过 streaming SSE 返回 batch 结果；每个 event 使用 `index` 标明所属请求。
5. Client 按 `index` 分流 event，计算每条请求的 TTFT、TPOT、E2E latency 和 speculative decoding 指标。
6. Observability collector 同期采集两个 server 的 `/v1/loads`，形成服务状态时间序列。

这里的 `batch.size` 表示单次 `/generate` 请求包含的样本数。例如 `batch.size=4` 会产生一个包含 4 组 `input_ids` 的 HTTP 请求。

## 2. 文件目录结构

```text
benchmark/decoupled_spec/
├── server-side/
│   ├── config.py
│   ├── verifier_server.py
│   └── drafter_server.py
├── client-side/
│   ├── client.py
│   ├── request_loader.py
│   └── metrics.py
├── plot/
│   ├── plot_latency.py
│   ├── plot_speculative.py
│   ├── plot_observability.py
│   ├── generate_report.py
│   ├── plot_utils.py
│   └── __init__.py
├── common/
│   ├── collector.py
│   ├── artifacts.py
│   └── __init__.py
├── configs/
│   ├── verifier/
│   │   └── tp1.yaml
│   ├── drafter/
│   │   └── tp1.yaml
│   ├── client/
│   │   ├── gsm8k.yaml
│   │   └── synthetic.yaml
│   └── observability/
│       └── default.yaml
├── skills/
│   ├── run-decoupled-spec-benchmark/
│   ├── operate-decoupled-spec-servers/
│   ├── send-decoupled-spec-workload/
│   ├── observe-decoupled-spec-run/
│   ├── analyze-decoupled-spec-results/
│   └── audit-decoupled-spec-artifacts/
├── README.md
└── __init__.py
```

### 2.1 `server-side/`

- `verifier_server.py`：解析 verifier YAML，设置该进程的 GPU 和环境变量，构造 SGLang `ServerArgs`，启动 verifier HTTP server，并记录进程状态。
- `drafter_server.py`：以相同方式启动 drafter HTTP server，并记录 drafter 的最终配置与状态。
- `config.py`：统一完成 YAML 读取、明确命名的 CLI 参数覆盖、`ServerArgs` 字段校验以及 verifier/drafter 配对校验。

配对校验会检查 role-specific algorithm、K/F/verify-window、Python/C++ data-plane backend、bind/connect endpoint 和 GPU 分配，使两个 role 在进入模型加载前就具有一致的通信配置。

### 2.2 `client-side/`

- `request_loader.py`：读取 dataset、选择样本、渲染 chat template、执行 tokenization，并生成确定的 `RequestSpec` 列表。
- `client.py`：构造一个 batch payload，发送 streaming `/generate` 请求，按 `index` 聚合 SSE event，并保存请求与响应记录。
- `metrics.py`：从 client 收集的请求记录中计算请求级和 batch 级指标，输出 CSV 与 JSON summary。

Client 加载的是 target tokenizer。Verifier 收到预先生成的 `List[List[int]]`，从而让输入文本、chat template、token IDs 和实际 `prompt_len` 都成为 benchmark 产物的一部分。

### 2.3 `common/`

- `collector.py`：启动时保存 `/model_info` 和 `/server_info`，运行期间按固定周期并行采集两个 server 的 `/v1/loads`。它只负责原始数据采集和 summary，不在退出路径中触发绘图。
- `artifacts.py`：初始化、记录和封存一次实验，维护 provenance、组件状态、run manifest 与 checksum。

Collector 默认每 1 秒采样一次。该时间序列适合观察整个正式请求窗口内的负载、队列和吞吐变化；CUDA stream、IPC 和单轮 verify 的微秒级分析可以再与 Nsys/NVTX trace 对齐。

### 2.4 `plot/`

- `plot_latency.py`：从 request-level CSV 生成 TTFT、TPOT 和 E2E latency 图。
- `plot_speculative.py`：从 request-level CSV 生成 accept rate 和 accept length 图。
- `plot_observability.py`：从原始 service samples 生成 queue、throughput、KV usage 和 speculative overview 图。
- `generate_report.py`：读取本轮 summary、resolved config、provenance 和前述 manifests，生成 Markdown 结果报告。
- `plot_utils.py`：保存统一的白底绘图风格、颜色、数据加载、图片导出和 source hash 工具。

这些脚本只消费一个 `RUN_DIR` 内已保存的 benchmark 产物，不发送请求，也不比较不同 run。每个脚本独立写入 manifest，记录自己的输入 SHA-256 和输出路径。

### 2.5 `common/artifacts.py`

`artifacts.py` 是整套 benchmark 的实验档案组件。它贯穿一次运行的三个阶段：

1. **初始化**：`init` 创建唯一的 `RUN_DIR`，记录时间、命令行、Python、平台、Git commit、branch 和工作区状态。
2. **运行中记录**：server、client 和 collector 调用 `write_json()` 与 `update_status()`，把解析后的配置和当前进程状态原子写入同一个实验目录。
3. **结束后封存**：`seal` 汇总 JSON 产物到 `run_manifest.json`，并为所有文件生成 `SHA256SUMS`。

因此，`artifacts.py` 管理的是“这轮实验是谁、使用了什么配置、产生了哪些文件、文件是否保持一致”；具体性能指标由 `client-side/metrics.py` 计算。

### 2.6 `configs/` 与 `skills/`

`configs/` 按 verifier、drafter、client 和 observability 四类组件保存 YAML。每个进程启动时都会保存应用命令行覆盖后的 resolved config。

`skills/` 保存六个面向不同任务的 Agent skill：

| Skill | 职责 |
| --- | --- |
| `run-decoupled-spec-benchmark` | 编排一轮完整 benchmark 的生命周期，不引入一键运行脚本 |
| `operate-decoupled-spec-servers` | 校验拓扑，独立启动、检查和停止 verifier/drafter |
| `send-decoupled-spec-workload` | 检查 dataset/tokenizer/chat template，并提交一个 streaming batch |
| `observe-decoupled-spec-run` | 采集并验收 verifier/drafter 的 service-level 时间序列 |
| `analyze-decoupled-spec-results` | 基于一个已有 run 的稳定产物生成图和报告 |
| `audit-decoupled-spec-artifacts` | 在封存前后检查产物完整性、一致性和 checksum |

每个 skill 都是独立 package，包含带 YAML frontmatter 的 `SKILL.md`、`agents/openai.yaml`，以及该任务确实需要的 `references/` 或确定性 `scripts/`。仓库通过 `.claude/skills/<skill-name>` 相对链接发现这些 benchmark-local package；`.codex/skills` 已统一暴露 `.claude/skills`。

## 3. 配置说明

### 3.1 Verifier 与 drafter

默认配置对应以下单机拓扑：

| Role | Model | TP | GPU | HTTP | Decoupled endpoint |
| --- | --- | ---: | --- | --- | --- |
| Verifier | Qwen3.5-27B | 1 | `0` | `127.0.0.1:30000` | bind `tcp://127.0.0.1:31000` |
| Drafter | Qwen3.5-0.8B | 1 | `1` | `127.0.0.1:30001` | bind `tcp://127.0.0.1:31001` |

每个 role 配置包含两类字段：

```yaml
schema_version: 1
role: verifier
runtime:
  cuda_visible_devices: ["0"]
  env:
    SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND: "1"
server_args:
  model_path: /path/to/target-model
  tp_size: 1
  host: 127.0.0.1
  port: 30000
  enable_metrics: true
  speculative_algorithm: DECOUPLED_VERIFY
  speculative_num_steps: 3
  speculative_eagle_topk: 1
  speculative_num_draft_tokens: 4
  decoupled_spec_role: verifier
  decoupled_spec_rank: 0
  decoupled_spec_bind_endpoint: tcp://127.0.0.1:31000
  decoupled_spec_connect_endpoints: [tcp://127.0.0.1:31001]
```

- `runtime` 控制进程级 GPU 可见性和环境变量。
- `server_args` 在字段校验后直接构造当前 checkout 的 SGLang `ServerArgs`。
- `enable_metrics: true` 为 `/v1/loads` observability 数据提供 server metrics。
- Verifier 使用 `DECOUPLED_VERIFY`；drafter 是普通 decode engine，
  `speculative_algorithm` 必须为 `null`，并显式禁用 overlap schedule。
- 第一阶段固定 F=1，即 `speculative_eagle_topk: 1`；K=3 时 verify window
  `speculative_num_draft_tokens` 必须为 4。
- Drafter 使用 `skip_server_warmup: true`，因为它只处理 decoupled control
  traffic，不接收通用 startup warmup 发送的 user `/generate` 请求。
- 两端的 bind endpoint 会分别出现在对端的 connect endpoint 列表中。
- Active decoupled runtime 强制两端同时设置
  `runtime.env.SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND=1`；role validation 会拒绝
  Python data plane，pair validator 也会拒绝两端 backend 混配。Python 实现只保留为
  CPU semantic/differential reference。

### 3.2 Client 与 dataset

Client 配置同时定义 verifier 地址、target tokenizer、batch、dataset、chat template 和 generation 参数：

```yaml
schema_version: 1
server:
  base_url: http://127.0.0.1:30000
target_tokenizer:
  model_path: /path/to/target-model
batch:
  size: 1
dataset:
  format: gsm8k
  path: /path/to/gsm8k
  seed: 0
  shuffle: false
  prompt_column: question
  reference_column: answer
chat_template:
  mode: tokenizer
  enable_thinking: false
generation:
  output_len: 1024
  temperature: 0
  ignore_eos: true
```

当前 loader 支持以下数据源：

| `dataset.format` | 数据来源 | Prompt/reference 选择方式 |
| --- | --- | --- |
| `gsm8k` | Parquet 文件或目录 | 默认读取 `question` / `answer`；目录内有 test 文件时优先使用 test split |
| `parquet`、`generic_parquet` | Parquet 文件或目录 | 通过 `prompt_column` / `reference_column` 指定字段 |
| `dapo_math_17k` | DAPO parquet 文件或目录 | 原生读取 `prompt` message list；reference 默认读取 `reward_model.ground_truth` |
| `jsonl`、`generic_jsonl`、`codeforces_raw`、`sharegpt` | JSONL 文件或目录 | 通过 `prompt_column` / `reference_column` 指定字段 |
| `synthetic_ids` | 配置生成 | 使用 `prompt_len`、`token_id` 和 `output_len` 构造固定长度请求 |

Chat template 有两种模式：

- `mode: tokenizer`：调用 target tokenizer 的 `apply_chat_template()`，并添加 generation prompt；Qwen tokenizer 可通过 `enable_thinking` 控制 thinking template。
- `mode: none`：直接 tokenize dataset 中的原始 prompt。

`generation.output_len` 会写入每条请求的 `max_new_tokens`。`ignore_eos: true` 适合固定输出长度实验，`ignore_eos: false` 适合保留数据集上的正常停止行为。

### 3.3 Observability

```yaml
schema_version: 1
interval_s: 1.0
request_timeout_s: 0.8
targets:
  verifier:
    base_url: http://127.0.0.1:30000
  drafter:
    base_url: http://127.0.0.1:30001
loads:
  include: [core, spec, queues]
```

`interval_s` 控制采样周期，`loads.include` 控制 `/v1/loads` 返回的字段组。Collector 对 verifier 和 drafter 并行发起采样请求，并记录每次请求的时间、HTTP status、采集延迟和原始 payload。

### 3.4 命令行覆盖

YAML 保存完整配置，常用实验变量通过明确命名的 CLI 参数覆盖。只有命令行中实际出现的参数会覆盖 YAML，最终结果仍会写入 resolved config。

| 组件 | 可覆盖参数 |
| --- | --- |
| Verifier/drafter | `--model-path`、`--tp-size`、`--cuda-visible-devices`、`--host`、`--port` |
| Client | `--base-url`、`--target-tokenizer-path`、`--batch-size`、dataset/chat-template/generation 相关参数 |
| Observability | `--verifier-url`、`--drafter-url`、`--interval-s`、`--request-timeout-s`、`--loads-include` |

例如：

```bash
# Verifier TP4，并分配 4 张 GPU
--tp-size 4 \
--cuda-visible-devices 0 1 2 3

# Drafter 使用另一张 GPU
--cuda-visible-devices 4

# Client 使用 BS4 和 1024-token 输出
--batch-size 4 \
--output-len 1024

# Observability 改为 0.5 秒采样
--interval-s 0.5
```

Boolean 参数同时提供正反形式，例如 `--ignore-eos` / `--no-ignore-eos` 和 `--enable-thinking` / `--no-enable-thinking`。

Speculative K、F、verify window、data-plane backend 和 bind/connect endpoints 继续作为 verifier/drafter 成对 YAML 配置，由 pair validator 一次检查两端的一致性。

v0.5.17 的 Qwen3.5-27B TP4 / Qwen3.5-0.8B TP1、K3/F1、DAPO row0
正式 overlap/non-overlap 配置与原子命令见
[`configs/formal_v0517_qwen35.md`](configs/formal_v0517_qwen35.md)。

BS8/16/32/64 × output 1K/4K/16K/32K、thinking + verifier ReplaySSM 的
32-case 扩展 campaign 见 [`matrix/README.md`](matrix/README.md)。该 campaign
仍复用本目录的四个独立 role；matrix 层只负责确定性 case contract、可恢复 ledger
和 sealed-only 汇总，不提供 combined launcher。当前 GPU matrix 被
`draft_inbox_segment_bs8_correctness` 硬前置 gate 阻挡，不能在迁移并完成真实 BS8
probe 之前宣称 ready 或开始实验。

## 4. 运行方式

下面给出一轮完整运行。所有命令均从 SGLang 仓库根目录执行，`PYTHONPATH=python` 确保使用当前 checkout 的源码。

Agent 执行完整实验时使用 `$run-decoupled-spec-benchmark`。如果只需要处理某一个阶段，可以直接使用对应的 server、workload、observability、analysis 或 artifact-audit skill。总控 skill 只负责协调下列独立命令和进程，不会调用隐藏的一键 runner。

### 4.1 校验配置

先校验 verifier 与 drafter 的配置配对关系：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server-side/config.py validate \
  --verifier-config benchmark/decoupled_spec/configs/verifier/tp1.yaml \
  --drafter-config benchmark/decoupled_spec/configs/drafter/tp1.yaml
```

当 server 使用 CLI 覆盖部署参数时，pair validator 接受带 role 前缀的同名参数。例如 verifier TP4：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server-side/config.py validate \
  --verifier-config benchmark/decoupled_spec/configs/verifier/tp1.yaml \
  --drafter-config benchmark/decoupled_spec/configs/drafter/tp1.yaml \
  --verifier-tp-size 4 \
  --verifier-cuda-visible-devices 0 1 2 3 \
  --drafter-cuda-visible-devices 4
```

检查 client YAML 的解析结果：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/client-side/client.py \
  --config benchmark/decoupled_spec/configs/client/gsm8k.yaml \
  --check
```

### 4.2 创建实验目录

```bash
RUN_DIR="$(python benchmark/decoupled_spec/common/artifacts.py init \
  --output-root /tmp/decoupled-spec-runs \
  --name qwen35-gsm8k-bs1)"

echo "$RUN_DIR"
```

后续四个组件使用同一个 `RUN_DIR`，这样所有输入、状态和结果会自然归入同一轮实验。

### 4.3 启动 verifier server

在一个长期进程会话中运行：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server-side/verifier_server.py \
  --config benchmark/decoupled_spec/configs/verifier/tp1.yaml \
  --run-dir "$RUN_DIR"
```

### 4.4 启动 drafter server

在另一个长期进程会话中运行：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server-side/drafter_server.py \
  --config benchmark/decoupled_spec/configs/drafter/tp1.yaml \
  --run-dir "$RUN_DIR"
```

Agent 或进程管理器分别持有两个 server 会话，可以独立查看日志、检查状态和采集 profiler trace。

两个 server 的 `status.json` 均进入 `http_ready` 后，即可开始正式采集：

```bash
cat "$RUN_DIR/roles/verifier/status.json"
cat "$RUN_DIR/roles/drafter/status.json"

curl -fsS http://127.0.0.1:30000/health
curl -fsS http://127.0.0.1:30001/model_info
```

### 4.5 启动 observability collector

在正式 client 请求前启动 collector：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/common/collector.py \
  --config benchmark/decoupled_spec/configs/observability/default.yaml \
  --run-dir "$RUN_DIR"
```

也可以使用 `--duration-s` 让 collector 在固定时间后结束，例如：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/common/collector.py \
  --config benchmark/decoupled_spec/configs/observability/default.yaml \
  --run-dir "$RUN_DIR" \
  --duration-s 120
```

### 4.6 发送一个正式 batch

GSM8K 配置：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/client-side/client.py \
  --config benchmark/decoupled_spec/configs/client/gsm8k.yaml \
  --run-dir "$RUN_DIR" \
  --batch-size 1 \
  --output-len 1024
```

固定 1K input / 1K output 配置：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/client-side/client.py \
  --config benchmark/decoupled_spec/configs/client/synthetic.yaml \
  --run-dir "$RUN_DIR"
```

Client 会在标准输出打印 batch summary，同时把完整记录写入 `$RUN_DIR/client/`。

### 4.7 结束采集并生成单次运行报告

Client 完成后，保留至少一轮 trailing sample，再向 collector 发送 SIGINT 或 SIGTERM。Collector 只在原始采集结束后写入 summary 和最终状态。随后由 Agent 结束两个 server 进程，并按顺序显式生成本轮的派生产物：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/plot/plot_latency.py \
  --run-dir "$RUN_DIR"

PYTHONPATH=python python benchmark/decoupled_spec/plot/plot_speculative.py \
  --run-dir "$RUN_DIR"

PYTHONPATH=python python benchmark/decoupled_spec/plot/plot_observability.py \
  --run-dir "$RUN_DIR"

PYTHONPATH=python python benchmark/decoupled_spec/plot/generate_report.py \
  --run-dir "$RUN_DIR"
```

四个脚本职责独立，分别写入自己的 source manifest。某一类派生产物失败时，可以只定位和重跑对应脚本，不会重新发送 benchmark 流量。

### 4.8 验收并封存本轮实验

先执行 pre-seal audit。该检查会核对组件状态、batch cardinality、formal window、observability coverage 和绘图 source hash，并把报告写入本轮产物：

```bash
python benchmark/decoupled_spec/skills/audit-decoupled-spec-artifacts/scripts/audit_run.py \
  --run-dir "$RUN_DIR" \
  --phase pre-seal \
  --output "$RUN_DIR/audit/pre_seal.json"
```

只有 pre-seal audit 通过后才封存：

```bash
python benchmark/decoupled_spec/common/artifacts.py seal \
  --run-dir "$RUN_DIR"

python benchmark/decoupled_spec/skills/audit-decoupled-spec-artifacts/scripts/audit_run.py \
  --run-dir "$RUN_DIR" \
  --phase sealed
```

封存完成后，`run_manifest.json` 提供集中索引，`SHA256SUMS` 用于检查每个产物文件的完整性。Sealed audit 是只读操作，同时检查 checksum 内容和封存后的实际文件集合；封存后不再向 `RUN_DIR` 写文件。

## 5. 指标口径

### 5.1 Latency 与 throughput

| 指标 | 计算方式 | 含义 |
| --- | --- | --- |
| `batch_elapsed_s` | batch 请求发出至整个 SSE stream 结束 | 单个 batch 的完整 HTTP streaming 时间 |
| `output_tokens_per_s` | `sum(completion_tokens) / batch_elapsed_s` | batch 级 output throughput |
| TTFT | batch 请求开始至该请求首次 token 数增加的 SSE event | 每条请求的 time to first token |
| TPOT | `(last_token_time - first_token_time) / (completion_tokens - 1)` | 每条请求首 token 之后的平均 per-output-token time |
| E2E latency | batch 请求开始至该请求最后一次 token 数增加 | 每条请求的端到端生成时间 |

TTFT、TPOT 和 E2E latency 都输出 `mean`、`p50`、`p95`、`p99`。这些值来自 client 观察到的 HTTP streaming 时间，覆盖 server 处理、网络传输和 client SSE 解析。`1000 / mean_tpot_ms` 可以作为单请求稳定生成阶段的近似 tokens/s；`output_tokens_per_s` 则直接表示整个 batch 的实际 output throughput。

Speculative decoding 一轮可能接受多个 token，同一 SSE event 也可能让 `completion_tokens` 增加多个。Client 使用 event 时间戳和累计 token 数计算 TPOT，使指标保持 token 口径。

### 5.2 Speculative decoding

| 指标 | 来源或公式 |
| --- | --- |
| `spec_verify_ct` | Verifier 返回的 verify 次数之和 |
| `spec_num_proposed_drafts` | 实际送入 verifier 的 draft token 数之和 |
| `spec_num_correct_drafts` | 通过 verifier 的 draft token 数之和 |
| `spec_accept_rate` | `correct_drafts / proposed_drafts` |
| `spec_draft_occupancy_rate` | `actual_proposed / (verify_ct * K)`，表示 draft 供给占 nominal verify capacity 的比例 |
| `spec_proposed_draft_length` | `actual_proposed / verify_ct`，表示每个 verify request-row 实际验证的 draft 数 |
| `spec_accept_length` | `completion_tokens / verify_ct` |

请求级原始值写入 `request_metrics.csv` 和 `responses.jsonl`，batch 聚合值写入 `summary.json`。

### 5.3 Decode 窗口时序

引擎每 `decode_log_interval` 个 decode iteration 冻结一个窗口，默认窗口为 40 步。`/v1/loads` 的 `decode_metrics_windows` 同时给出该窗口的：

- `iter_latency_ms`：窗口 elapsed time 除以 decode iteration 数；
- `mean_batch_size`：窗口内 decode request-row 数除以 iteration 数；
- `mean_context_length`：窗口内所有 decode request-row 的 context length 均值；
- `proposed_draft_length`：实际送入 verifier 的 draft 数除以 verify request-row 数，即图中的 valid draft length；
- `accept_length`：接受的 draft 与 bonus token 总数除以 verify request-row 数。

窗口带单调 `window_id` 和 `end_time`。Collector 只保存 HTTP 原始响应；`plot_observability.py` 离线去重窗口并绘制共享时间轴的 scheduler cycle、mean batch size、mean context length、valid draft length 和 accept length。该链路不解析 server log，也不增加 GPU 同步。

Collector 的 `observability/summary.json` 会分别汇总 verifier 和 drafter，给出各自的窗口数、scheduler cycle mean/min/p50/p95/max、mean batch size 和 mean context length；verifier 还包含 valid draft length 与 accept length。重复出现在多个 HTTP sample 中的 bounded-history 窗口只统计一次。

## 6. 输出产物目录

一轮完整实验的产物结构如下：

```text
$RUN_DIR/
├── provenance/
│   └── run_start.json
├── logs/
│   ├── verifier.log
│   └── drafter.log
├── roles/
│   ├── verifier/
│   │   ├── resolved_config.json
│   │   └── status.json
│   ├── drafter/
│   │   ├── resolved_config.json
│   │   └── status.json
│   ├── client/
│   │   └── status.json
│   └── observability/
│       └── status.json
├── client/
│   ├── resolved_config.json
│   ├── verifier_model_info.json
│   ├── sampled_requests.jsonl
│   ├── formal_window.json
│   ├── stream_timing_events.jsonl
│   ├── raw_batch_response.json
│   ├── responses.jsonl
│   ├── request_metrics.csv
│   └── summary.json
├── observability/
│   ├── resolved_config.json
│   ├── startup/
│   │   ├── verifier/
│   │   │   ├── model_info.json
│   │   │   └── server_info.json
│   │   └── drafter/
│   │       ├── model_info.json
│   │       └── server_info.json
│   ├── samples.jsonl
│   ├── summary.json
│   └── plots/
│       ├── decode_metrics.svg
│       ├── decode_metrics.png
│       ├── overview.svg
│       ├── overview.png
│       └── plot_manifest.json
├── plots/
│   ├── request_latency.svg
│   ├── request_latency.png
│   ├── request_latency_manifest.json
│   ├── request_speculative.svg
│   ├── request_speculative.png
│   ├── request_speculative_manifest.json
│   ├── run_report.md
│   └── run_report_manifest.json
├── audit/
│   └── pre_seal.json
├── run_manifest.json
└── SHA256SUMS
```

### 6.1 实验身份与实际配置

- `provenance/run_start.json`：保存实验名称、启动时间、命令行、Python 路径、平台、Git commit、branch 和初始工作区状态。
- `roles/*/resolved_config.json`：保存 verifier 和 drafter 实际使用的 role 配置。
- `client/resolved_config.json`、`observability/resolved_config.json`：保存 client 和 collector 应用明确 CLI 覆盖后的最终配置。
- `roles/*/status.json`：保存各组件的当前状态、PID、更新时间以及完成或失败信息。

### 6.2 Client 输入与 streaming 原始记录

- `sampled_requests.jsonl`：逐请求保存 dataset 行号、原始 prompt、渲染后的 prompt、`input_ids`、`prompt_len`、期望输出长度和 reference response。
- `verifier_model_info.json`：保存正式请求前读取到的 verifier model 信息。
- `formal_window.json`：标记正式 batch 的开始、结束和 elapsed time；observability 图会用该窗口标出 benchmark 区间。
- `stream_timing_events.jsonl`：逐 SSE event 保存接收时间、batch `index`、request ID、累计 completion tokens 和 finish reason。
- `raw_batch_response.json`：保存每个 batch index 收到的最后一个完整 response event。
- `responses.jsonl`：将模型输出、reference response、latency 和 speculative 指标整理成逐请求记录。

### 6.3 指标结果

- `request_metrics.csv`：适合逐请求筛选和后续统计分析。
- `summary.json`：保存 batch size、token 数、throughput、TTFT/TPOT/E2E 分位数以及 speculative decoding 聚合指标。

### 6.4 Observability 时间序列

- `startup/<role>/model_info.json`、`server_info.json`：保存 collector 启动时的服务快照与采集元信息。
- `samples.jsonl`：逐采样、逐 role 保存 `/v1/loads` 原始 payload、采集时间、HTTP status 和采集延迟。
- `summary.json`：保存采样周期、样本数、目标数和错误数。
- `observability/plots/overview.svg`、`overview.png`：从 `samples.jsonl` 派生的可视化。
- `observability/plots/plot_manifest.json`：记录 overview 的输入文件、SHA-256 和输出路径。

重新生成图表：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/plot/plot_observability.py \
  --run-dir "$RUN_DIR"
```

### 6.5 Benchmark 结果呈现

- `plots/request_latency.svg`、`.png`：逐请求展示 TTFT、TPOT 和 E2E latency 原始值。
- `plots/request_speculative.svg`、`.png`：逐请求展示 accept rate 和 accept length；当 response 中没有 speculative 指标时，该图不会生成。
- `plots/run_report.md`：汇总 target/draft 配置、batch/dataset 参数、throughput、latency mean/p50/p95/p99 和 speculative decoding 指标。
- `plots/request_latency_manifest.json`：记录 latency 图的输入、SHA-256 和输出路径。
- `plots/request_speculative_manifest.json`：记录 speculative 图的输入、SHA-256 和输出路径；没有可用 speculative 字段时，manifest 仍会生成且 outputs 为空。
- `plots/run_report_manifest.json`：记录 Markdown 报告使用的 summary、配置、provenance 和绘图 manifests。

### 6.6 封存结果

- `logs/verifier.log`、`logs/drafter.log`：Agent 独立启动两个 server 时保存的 stdout/stderr，用于解释启动失败和最终退出状态。
- `audit/pre_seal.json`：封存前对状态、请求 cardinality、observability coverage 和 plot provenance 的机器可读验收结果。
- `run_manifest.json`：集中收录本轮所有 JSON 产物，便于一次性检查配置、状态与结果。
- `SHA256SUMS`：记录每个产物文件的 SHA-256，可用于传输后或长期保存后的完整性验证。

建议按以下顺序阅读一轮结果：

1. `plots/run_report.md`：先看配置与主要结果的集中摘要。
2. `plots/request_latency.svg` 和 `request_speculative.svg`：查看逐请求分布。
3. `client/summary.json` 与 `client/request_metrics.csv`：检查聚合值和原始请求指标。
4. `observability/plots/overview.svg`：将正式请求窗口与 queue、throughput、KV usage 对齐。
5. `client/stream_timing_events.jsonl` 与 `observability/samples.jsonl`：检查原始时序。
6. resolved config、provenance、各派生产物 manifest 和 `SHA256SUMS`：确认实验输入与产物完整性。
