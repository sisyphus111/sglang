# v0.5.17 Qwen3.5 Decoupled-Spec 正式实验配置

这两轮实验共享同一个 drafter、client 和 observability 配置，只替换
verifier 配置中的 `disable_overlap_schedule`。这样 overlap/non-overlap 的
对照轴是显式且唯一的；四个 runtime role 仍由 Agent 独立启动，不存在
combined launcher。

## 实验 tuple

| 轴 | Non-overlap | Overlap |
| --- | --- | --- |
| Target | Qwen3.5-27B, TP4, GPU 0-3 | 相同 |
| Drafter | Qwen3.5-0.8B, TP1, GPU 4 | 相同 |
| Draft shape | K=3, F=1 (`topk=1`), verify window=4 | 相同 |
| Data plane | C++ pybind | C++ pybind |
| Dataset | DAPO-Math-17k row 0, tokenizer chat template, thinking | 相同 |
| Request | BS1, output=1024, temperature=0, ignore EOS | 相同 |
| Verifier schedule | `disable_overlap_schedule=true` | `disable_overlap_schedule=false` |

配置文件：

```bash
DRAFT_CONFIG=benchmark/decoupled_spec/configs/drafter/qwen35_0_8b_tp1_k3_f1_cpp.yaml
CLIENT_CONFIG=benchmark/decoupled_spec/configs/client/dapo_math_17k_row0_thinking_bs1_out1024.yaml
OBS_CONFIG=benchmark/decoupled_spec/configs/observability/qwen35_tp4_tp1.yaml
NON_OVERLAP_CONFIG=benchmark/decoupled_spec/configs/verifier/qwen35_27b_tp4_k3_f1_non_overlap_cpp.yaml
OVERLAP_CONFIG=benchmark/decoupled_spec/configs/verifier/qwen35_27b_tp4_k3_f1_overlap_cpp.yaml
```

Active runtime 要求两端都设置 `SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND=1`；Python
data plane 只作为 CPU semantic/differential reference。Verifier 不设置固定时长的
snapshot wait：draft token 通过 ZMQ 流式到达后，由 verifier daemon 异步落入 GPU
tail buffer；每轮 verify 在自己的 stream 上直接从 GPU tail 取得 snapshot。Daemon 的
CPU mirror 继续处理 general pending realignment；GPU 每个 request 只发布当前唯一的
consumable linear tail，pending 时发布空/不可消费状态，不保存多个 round。Pair validator
会拒绝 Python/C++ data-plane 混配；drafter 保持普通 decode engine，
`speculative_algorithm` 必须为 `null`。

## 先做不加载模型的检查

从目标 worktree 根目录运行，并确保 `PYTHONPATH=python` 指向该 checkout：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server-side/config.py validate \
  --verifier-config "$NON_OVERLAP_CONFIG" \
  --drafter-config "$DRAFT_CONFIG"

PYTHONPATH=python python benchmark/decoupled_spec/server-side/config.py validate \
  --verifier-config "$OVERLAP_CONFIG" \
  --drafter-config "$DRAFT_CONFIG"

PYTHONPATH=python python benchmark/decoupled_spec/server-side/verifier_server.py \
  --config "$NON_OVERLAP_CONFIG" --run-dir /tmp/decoupled-spec-config-check --check

PYTHONPATH=python python benchmark/decoupled_spec/server-side/verifier_server.py \
  --config "$OVERLAP_CONFIG" --run-dir /tmp/decoupled-spec-config-check --check

PYTHONPATH=python python benchmark/decoupled_spec/server-side/drafter_server.py \
  --config "$DRAFT_CONFIG" --run-dir /tmp/decoupled-spec-config-check --check

PYTHONPATH=python python benchmark/decoupled_spec/client-side/client.py \
  --config "$CLIENT_CONFIG" --check

PYTHONPATH=python python \
  benchmark/decoupled_spec/skills/send-decoupled-spec-workload/scripts/inspect_workload.py \
  --config "$CLIENT_CONFIG"

PYTHONPATH=python python benchmark/decoupled_spec/common/collector.py \
  --config "$OBS_CONFIG" --run-dir /tmp/decoupled-spec-config-check --check
```

检查结果必须解析到实际模型与 dataset 路径、TP4/TP1、K3/F1、BS1、thinking、
1024 输出；不能用 CLI override 暗中改变这些轴。

当前 checkout 的只读 materialization 检查已确认：选中 dataset row 0，
`dataset_index=9a9b6eb4-a1cb-49d1-8c1e-62eaf2f74079`，prompt length 为
199，`input_ids` SHA-256 为
`f9d5c44b2268257a42c192414359c1485441d35e25e10f00d0dc6f4601ddfa8d`。
正式运行前应重新检查这些值；变化意味着 tokenizer、chat template 或 dataset
provenance 已改变，不能与原 tuple 混报。

## 每轮实验的原子执行顺序

以下命令每次只选择一个 verifier 配置。Non-overlap 使用
`$NON_OVERLAP_CONFIG`，overlap 使用 `$OVERLAP_CONFIG`，并为两轮创建不同的
`RUN_DIR`。

```bash
VERIFY_CONFIG="$NON_OVERLAP_CONFIG"  # overlap 轮替换为 $OVERLAP_CONFIG
RUN_NAME=qwen35-27b-tp4-draft-0.8b-tp1-k3-f1-bs1-dapo-thinking-out1024-nonoverlap-cpp
RUN_DIR="$(PYTHONPATH=python python benchmark/decoupled_spec/common/artifacts.py init \
  --output-root benchmark/decoupled_spec/results \
  --name "$RUN_NAME")"
mkdir -p "$RUN_DIR/logs"
```

在两个独立的长期会话中启动 server，并分别保存日志：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server-side/verifier_server.py \
  --config "$VERIFY_CONFIG" --run-dir "$RUN_DIR" \
  >"$RUN_DIR/logs/verifier.log" 2>&1
```

```bash
PYTHONPATH=python python benchmark/decoupled_spec/server-side/drafter_server.py \
  --config "$DRAFT_CONFIG" --run-dir "$RUN_DIR" \
  >"$RUN_DIR/logs/drafter.log" 2>&1
```

只有 deterministic readiness gate 通过后才能继续：

```bash
PYTHONPATH=python python \
  benchmark/decoupled_spec/skills/operate-decoupled-spec-servers/scripts/wait_for_roles.py \
  --run-dir "$RUN_DIR" --timeout-s 600
```

随后在第三个独立会话启动 collector。等 verifier 和 drafter 都至少有一个
成功的 baseline `/v1/loads` sample，再执行一次正式 client：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/common/collector.py \
  --config "$OBS_CONFIG" --run-dir "$RUN_DIR" \
  >"$RUN_DIR/logs/collector.log" 2>&1
```

```bash
PYTHONPATH=python python benchmark/decoupled_spec/client-side/client.py \
  --config "$CLIENT_CONFIG" --run-dir "$RUN_DIR" \
  >"$RUN_DIR/logs/client.log" 2>&1
```

Client 完成后至少保留一个 sampling interval，确认两端都有 trailing sample，
再优雅结束 collector。只停止本轮持有的 verifier/drafter 会话或 PID，不使用
进程名级 `pkill`。

## 派生产物与封存门槛

Server 和 collector 全部退出后，依次运行：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/skills/observe-decoupled-spec-run/scripts/validate_samples.py \
  --run-dir "$RUN_DIR"
PYTHONPATH=python python benchmark/decoupled_spec/plot/plot_latency.py --run-dir "$RUN_DIR"
PYTHONPATH=python python benchmark/decoupled_spec/plot/plot_speculative.py --run-dir "$RUN_DIR"
PYTHONPATH=python python benchmark/decoupled_spec/plot/plot_observability.py --run-dir "$RUN_DIR"
PYTHONPATH=python python benchmark/decoupled_spec/plot/generate_report.py --run-dir "$RUN_DIR"
PYTHONPATH=python python benchmark/decoupled_spec/skills/audit-decoupled-spec-artifacts/scripts/audit_run.py \
  --run-dir "$RUN_DIR" --phase pre-seal --output "$RUN_DIR/audit/pre_seal.json"
```

只有 pre-seal audit 退出码为 0 才能封存：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/common/artifacts.py seal --run-dir "$RUN_DIR"
PYTHONPATH=python python benchmark/decoupled_spec/skills/audit-decoupled-spec-artifacts/scripts/audit_run.py \
  --run-dir "$RUN_DIR" --phase sealed
(cd "$RUN_DIR" && sha256sum --check SHA256SUMS)
```

正式结果至少要满足：client/collector 状态为 `completed`、BS1 cardinality
全链路一致、两端 observability 覆盖 baseline/formal/trailing、
`spec_verify_ct > 0`，并且 sealed audit 与 checksum 全部通过。
