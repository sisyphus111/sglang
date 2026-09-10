# Decoupled-Spec Benchmark

`benchmark/decoupled_spec/` 是一套可组合、可观测的 speculative decoding
benchmark，同时支持 coupled MTP/EAGLE 和 decoupled speculation。稳定源码只包含 server、client、observer、plot、可复用配置
和 Agent Skills；具体实验输入、运行结果和临时诊断不作为仓库接口。

## 1. 运行模型

一次标准实验由五个职责清晰的组件组成：

| 组件 | 入口 | 职责 |
| --- | --- | --- |
| Server fleet | `server/server.py` | 连接 Ray，放置 coupled target 或 decoupled verifier/drafter，自动分配端口并输出 ready manifest |
| Runner | `runner.py` | 启停 Client/Observer、等待 baseline/trailing barrier，并写跨组件时间线 |
| Client | `client/client.py` | 本地加载和 tokenize 数据，一次性向 target 或 verifier 提交整个 streaming batch |
| Observer | `client/observer.py` | 周期查询 manifest 中所有 engine 的只读 HTTP 指标并保存时间序列 |
| Plot | `plot/*.py` | 只读取已保存产物，生成图和 Markdown 报告，不重新发送流量 |

标准生命周期是：

```text
server ready
  -> observer baseline
  -> client formal batch
  -> observer trailing samples
  -> stop owned processes
  -> plots and report
```

`batch.size=N` 表示一个 `/generate` 请求中包含 `N` 个 tokenized inputs，不是
启动 N 个独立 HTTP 请求。Coupled 流量发给 target；decoupled 流量发给 verifier，
drafter 通过 decoupled-spec 控制面和数据面参与推理。

## 2. 目录结构

```text
benchmark/decoupled_spec/
├── server/                 # 统一 Ray fleet launcher 和 verifier profile 入口
├── client/                 # client、observer 和固定结果 schema
├── plot/                   # 标准单次运行图和报告
├── runner.py               # Client/Observer 编排和 benchmark 时间线
├── run_io.py               # 各组件共用的目录、JSON 和状态写入 helper
├── configs/
│   ├── server/             # 每个 YAML 都是完整 verifier+drafter fleet
│   ├── client/             # 可复用 workload 模板
│   └── observer/           # 可复用 HTTP sampling 模板
├── skills/                 # Decoupled-Spec Agent Skill 的唯一源码
├── analysis/               # 本地临时研究，不纳入提交
├── results/                # 运行与 study 产物，不纳入提交
└── README.md
```

Runner 位于 `runner.py`，Client 和 Observer 的独立入口仍位于 `client/`。最终数据写入调用方创建的 `RUN_DIR`；Server
manifest、状态和日志写入独立的临时 `RUNTIME_DIR`。公共 IO helper 位于
`run_io.py`。

## 3. 配置规则

### 3.1 Server

一次 deployment 只消费一个统一 YAML。Schema v1 保持 verifier/drafter 形式：

```yaml
schema_version: 1
ray:
  address: auto
  namespace: decoupled-spec-benchmark

verifier:
  replicas: 1
  runtime:
    env: {}
  server_args:
    model_path: /path/to/target
    tp_size: 4
    speculative_algorithm: DECOUPLED_VERIFY

drafter:
  replicas: 1
  runtime:
    env: {}
  server_args:
    model_path: /path/to/drafter
    tp_size: 1
    speculative_algorithm: null
```

不同模型、TP、K 或 schedule mode 可以使用不同的统一 YAML，但 launcher 不接受
独立 verifier/drafter 配置并在运行时拼接。Host、HTTP/NCCL/transport ports、rank、
GPU allocation 和 sparse peer configs 由 Ray launcher 生成，不写入模板。

Coupled MTP/EAGLE 使用 schema v2 和单一 target role：

```yaml
schema_version: 2
deployment: coupled_spec
ray:
  address: auto
  namespace: spec-benchmark
target:
  replicas: 1
  runtime:
    env: {}
  server_args:
    model_path: /path/to/target
    tp_size: 4
    speculative_algorithm: EAGLE
    speculative_draft_model_path: /path/to/target
    speculative_num_steps: 3
    speculative_eagle_topk: 1
    speculative_num_draft_tokens: 4
```

Target 仍由 Ray placement group 管理，但不创建 decoupled transport socket、peer
config 或 quota graph。MTP 返回 fixed-K acceptance histogram 时，Client 会转换为
固定结果 contract 使用的 per-position proposed/correct/rate 数组。

### 3.2 Decoupled 功能开关

Decoupled verifier 和 drafter 的执行行为都写在同一个 fleet YAML 中。例如：

```yaml
verifier:
  runtime:
    env:
      SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND: "1"
      SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL: "1"
      SGLANG_DECOUPLED_VERIFY_THROUGHPUT_PROFILE_PATH: /path/to/profile.json
  server_args:
    disable_overlap_schedule: false
    speculative_num_steps: 6
    speculative_eagle_topk: 1
    speculative_num_draft_tokens: 7
    speculative_adaptive: true
    speculative_adaptive_config: /path/to/adaptive.json

drafter:
  runtime:
    env:
      SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND: "1"
  server_args:
    disable_overlap_schedule: false
    speculative_num_steps: 6
    speculative_eagle_topk: 1
    speculative_num_draft_tokens: 7
```

- **Adaptive K**：`speculative_adaptive=true` 让 verifier 在配置的候选 K 中动态选择。
  正式实验必须先生成完整的 scheduler-cycle cost profile，并同时保存 profile、adaptive
  配置和 SHA-256。它调整的是 verifier 的 active K，不会在线改变模型、TP 或 fleet
  topology。
- **Partial draft tail**：`SGLANG_DECOUPLED_SPEC_ALLOW_PARTIAL` 默认为 `"1"`。
  设为 `"0"` 时，verifier 的 GPU selector 会等待所有 live request 都取得当前 active K
  个可消费 draft tokens，并等待 pending committed prefix 完成；snapshot shape 不变。
- **C++/Python transport**：`SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND="1"` 使用 C++
  threads/libzmq，`"0"` 使用 Python threads/pyzmq。两种 transport 使用相同的 wire
  codec 和 native GPU backend，允许 Python/C++ peer 混合部署；Python transport 仍需要
  编译 GPU extension。
- **Verifier/drafter overlap**：两个 role 分别设置 `disable_overlap_schedule`。两边都为
  `false` 即 dual overlap，也可以只开启一侧。当前 drafter overlap 要求 TP1、
  `page_size=1`、shared GPU backend、`disable_radix_cache=true`，并且不能启用 mixed
  chunked prefill 或 ReplaySSM。

Verifier 和 drafter 必须使用相同的 K、top-k 和 K+1 verify-token width。Adaptive 模式下，
YAML 中的 K 是最大捕获宽度，运行时 active K 只能从已配置且已 profile 的候选值中选择。

### 3.3 Client 和 Observer

Client YAML 定义 tokenizer、dataset、chat template、batch 和 generation 参数。
Observer YAML 定义采样周期、HTTP timeout 和 `/v1/loads` 字段组；正式运行时通过
`--server-manifest` 展开 target 或 verifier/drafter engine，静态 URL 只用于本地检查。

### 3.4 具体实验输入

`configs/experiments/` 和 `configs/matrix/` 是保留为 untracked 的本地运行输入区。正式运行前，
Campaign Skill 会把输入 YAML、所有引用配置、materialized cases、ledger 和 SHA-256
保存到 campaign 产物。可复现性以物化后的产物为准，不依赖工作树中的临时文件。

## 4. 单次实验命令

以下命令从仓库根目录执行。

手动创建唯一 `RUN_DIR` 和临时 `RUNTIME_DIR`：

```bash
RUN_DIR="$(pwd)/benchmark/decoupled_spec/results/my-decoupled-spec-run"
RUNTIME_DIR="$(mktemp -d)"
mkdir -p "$RUN_DIR" "$RUNTIME_DIR"
```

Server 使用 `RUNTIME_DIR` 保存控制文件；Client、Observer 和 Plot 只把最终数据写入
`RUN_DIR`。这些顶层目录都由调用方创建。

校验统一 fleet 配置，不连接 Ray：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server/server.py \
  --config benchmark/decoupled_spec/configs/server/<fleet>.yaml \
  --run-dir "$RUN_DIR" \
  --check
```

启动 fleet：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server/server.py \
  --config benchmark/decoupled_spec/configs/server/<fleet>.yaml \
  --run-dir "$RUN_DIR" \
  --runtime-dir "$RUNTIME_DIR"
```

Manifest ready 后由 Runner 启动 Observer、等待 baseline、执行 Client、保留 trailing
sample，再停止 Observer：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/runner.py \
  --client-config benchmark/decoupled_spec/configs/client/<workload>.yaml \
  --observer-config benchmark/decoupled_spec/configs/observer/default.yaml \
  --server-manifest "$RUNTIME_DIR/server/manifest.json" \
  --engine-rank 0 \
  --run-dir "$RUN_DIR"
```

单个 engine 时可以省略 rank。多个 coupled target 使用 `--engine-rank`；decoupled
case 继续支持 `--verifier-rank`。Runner 只在所有 engine 都有成功且
`num_waiting_reqs == 0` 的完整 baseline 轮次后
让 Client 打流量；Client 收齐 SSE 响应后，Runner 再等待一轮所有 target 的 trailing
sample，然后优雅停止 Observer。随后从保存的数据生成标准图和报告，一次实验即完成。通常应直接使用
`run-decoupled-spec-benchmark` Skill 协调整个生命周期。

## 5. Observer 与图

Observer 只周期查询 `/v1/loads`，不解析 server log，也不调用 `/generate`。Observer
拥有 `samples.jsonl`；跨组件的 `bench_timeline.json` 由 Runner 写入同一目录：

```text
observer/samples.jsonl
observer/bench_timeline.json
```

`bench_timeline.json` 只保存
`observer_started_wall_time → client_started_wall_time → client_finished_wall_time → observer_finished_wall_time`
四个边界，以及二者 Observer 边界之差 `observer_elapsed_s`。Baseline/trailing
sample 仍是 Runner 的执行 barrier，但不进入持久化 timeline。

标准时间序列以 Batch runtime 为横轴，包括：

- verifier/drafter iteration latency；
- verifier valid draft tail length 和 accept length；
- running batch size；
- token usage 和 queue 状态；
- drafter send queue latency exact mean；
- calibrated draft transport one-way latency exact mean；
- verifier receive-to-GPU-publish-enqueue latency exact mean；
- GPU publish completion latency、queue depth、frames/s 和 tokens/s（存在时）。

Latency 在每个短 decode window 内保存 histogram。通信图按 observer 轮询点归属首次看到的
engine windows，先合并这些 window 的原始 `sum_us` 和 `count`，再以
`sum(sum_us) / sum(count)` 画加权 exact mean。重复出现的 bounded-history window 不会再次
计入；该轮没有新 window 或合并后 count 为零时保留为空缺，不补零。跨节点 one-way
latency 使用记录事件时已经校准的样本。

## 6. 产物

一个完整 `RUN_DIR` 固定为：

```text
RUN_DIR/
├── config.json                         # effective server + client config
├── client/
│   ├── requests.csv
│   ├── batch.json
│   └── content.json
├── observer/
│   ├── samples.jsonl
│   └── bench_timeline.json
└── plots/
    ├── request_latency.png
    ├── request_speculative.png
    ├── overview.png
    ├── decode_metrics.png              # 有 decode windows 时生成
    ├── decoupled_spec_metrics.png      # 有 decoupled-spec windows 时生成
    ├── adaptive_verify.png             # Adaptive run 才生成
    └── run_report.md
```

`client/` 必须只包含上面三个文件。文件名、字段、顺序、单位和公式的
唯一 contract 位于
`skills/send-decoupled-spec-workload/references/client-artifact-contract.md`；除非用户明确
要求修改，否则不得扩展、重命名或恢复旧 client 结果文件。

HTTP readiness 本身不能证明 speculative path 生效。正式结果必须同时满足
`spec_verify_ct > 0`、请求 cardinality 完整，并且 Observer 覆盖 baseline、formal
window 和 trailing sample。Client 发流量前要求所有 engine 都有零 waiting queue 的
baseline。固定 BS 吞吐只检查首个请求退出前的满 BS decode measurement window：该窗口内
verifier 和 drafter 的 `num_waiting_reqs` 必须为零；prefill/batch-fill 和窗口结束后的 drain
阶段允许排队。缺少完整满 BS decode window 或窗口内 queue sample 时，不能把结果判为有效
吞吐数据。

`RUNTIME_DIR` 中的 manifest、status、resolved config 和 Server logs 只服务运行期，
不属于 benchmark 交付物，实验结束后可以删除。

## 7. Skills

所有 Decoupled-Spec benchmark Skill 的唯一可编辑源码位于：

```text
benchmark/decoupled_spec/skills/<skill-name>/
```

`.claude/skills/<skill-name>` 只是相对软链接。具体约束见 `skills/AGENTS.md`。

| Skill | 用途 |
| --- | --- |
| `operate-decoupled-spec-servers` | 校验、启动、检查和停止统一 fleet |
| `send-decoupled-spec-workload` | 检查并提交一个 streaming batch |
| `observe-decoupled-spec-run` | 采集和验证所有 engine 的 HTTP 时间序列 |
| `profile-decoupled-spec-verifier` | 生成和校验 adaptive verifier 使用的 cost profile |
| `run-decoupled-spec-benchmark` | 完成一个 case 的全部生命周期 |
| `run-decoupled-spec-campaign` | 展开、恢复、运行和汇总多个 case |
| `analyze-decoupled-spec-results` | 从一个已保存 RUN_DIR 生成标准图和报告 |

通常直接使用 `run-decoupled-spec-benchmark` 完成单个 case；它串联 fleet readiness、
Observer baseline、Client、trailing sample、停止 owned processes、校验、绘图和报告。
多个模型、K、schedule mode 或 batch size 使用 `run-decoupled-spec-campaign` 物化 case、
记录可恢复 ledger，并逐个执行相同生命周期。因此 Agent 可以从统一配置完成整个实验流程，
而不是只启动 server 或只发送 Client 请求。

## 8. 临时分析

`analysis/` 是本地 scratch 区，不是稳定 API；其中内容保持 untracked，不纳入提交。专题研究建议保存到：

```text
results/studies/<study-name>/
```

若临时脚本支撑正式结论，应在 study manifest 中记录脚本、输入和输出 SHA。通用逻辑
成熟后再晋升到 `plot/` 或 profiling Skill。
