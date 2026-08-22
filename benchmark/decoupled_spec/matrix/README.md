# Qwen3.5 Decoupled-Spec ReplaySSM 扩展矩阵

## 当前判断与 campaign 边界

最终正式 campaign 是
[`qwen35-dapo-replayssm-scaling-postslot-v5`](../results/campaigns/qwen35-dapo-replayssm-scaling-postslot-v5/ledger.json)，
contract SHA-256 为
`b3c86af070c9e1b4393d7d2294b1572ca2be95f858bf3bd1c4292323d68a475c`。
截至 2026-08-22，32/32 case 已 sealed + verified，0 active；完整结果、图和
面向阅读者的中文进展报告位于
[`summary/experiment_progress_report_zh.md`](../results/campaigns/qwen35-dapo-replayssm-scaling-postslot-v5/summary/experiment_progress_report_zh.md)；
更偏 profiler 和机制细节的版本位于
[`summary/performance_report.md`](../results/campaigns/qwen35-dapo-replayssm-scaling-postslot-v5/summary/performance_report.md)。

v5 有两个关键 correctness 边界：

- verifier 的 target 请求不等待 drafter prefill gate；开头可以 target-only
  推进，drafter 进入 decode 且 proposal 落到 GPU tail 后，后续 verify 自然开始
  消费 draft token；
- BS>1 性能矩阵不依赖 `enable_deterministic_inference` 的跨 run
  token-bitwise exact。初步 correctness 看内容是否可正常解码、无串请求/损坏，
  以及 speculative accounting、acceptance 和 occupancy 是否正常。

v5 还包含两项 profiler-driven 优化和一个扩展性修复：

- terminal marker 从 scalar advanced indexing 改成全 GPU `scatter_`，消除每轮约
  5.1--6.2 ms 的隐式 `cudaStreamSynchronize`；
- top-k=1 线性 `selected_index/parent_list` 按 request-pool capacity 一次性预分配，
  `verify_input_prepare` CPU p50 从 0.401 ms 降到 0.251 ms；
- continued chunked request 不再重复计入新 request-slot 预算。修复后所有 BS64
  case 从首轮 decode 起均为 `running=64, queue=0`，TTFT p99 为 522--598 ms；
  修复前第 64 条请求曾等待 15.9--427.9 s。

最终矩阵中 overlap 在 12/16 个 paired 点更快；16 个 case 总工作量的 wall time
减少 9.38%。但 overlap occupancy 在 16/16 个点都更低，output/verify 平均下降
10.8%；verify-row rate 则在 16/16 个点都更高，平均提高 17.1%。因此当前
overlap 的实现没有把 target graph 变慢，下一阶段瓶颈是 draft supply/tail
freshness，而不是再删除一个已知 host sync。

## 测什么，哪些条件保持不变

矩阵共有 32 个 case：

```text
schedule mode {nonoverlap, overlap}
  × batch size {8, 16, 32, 64}
  × max response length {1024, 4096, 16384, 32768}
```

所有 case 固定以下条件：

| 固定轴 | 值 |
| --- | --- |
| Target | Qwen3.5-27B, TP4, GPU 0-3 |
| Drafter | Qwen3.5-0.8B, TP1, GPU 4 |
| Spec shape | K=3, F=1, verify window=4 |
| Data plane | C++ + verifier GPU tail buffer |
| Dataset | DAPO-Math-17k，前 BS 条、`shuffle=false` |
| Template | target tokenizer，`enable_thinking=true` |
| Generation | temperature=0，ignore EOS |
| Server RNG | verifier/drafter 均固定 `random_seed=42` |
| Verifier state | `enable_linear_replayssm_spec=true` |
| Pool fairness | 两种 mode 都用 `extra_buffer`、320 Mamba slots、2.2M max tokens |
| Drafter rollback | 64 active requests、512 Mamba slots、2.2M max tokens |
| Decode CUDA Graph | 两端只捕获 request buckets 8/16/32/64 |

Verifier 不能同时启用普通 decode ReplaySSM：`enable_linear_replayssm=false` 与 `enable_linear_replayssm_spec=true` 是矩阵 contract。`SGLANG_RAGGED_VERIFY_MODE=static` 和 `SGLANG_DECOUPLED_SPEC_USE_CPP_PYBIND=1` 也由 materializer 强制检查。

`cuda_graph_bs_decode` 的单位是 request slots。Verifier 的 target-verify runner 会再乘固定 verify width=4，所以 BS64 对应 256 个 verify token rows，而不是把 capture bucket 写成 256；raw BS1–64 都 padding 到下一个 8/16/32/64 bucket。Drafter decode width=1，使用同一组 request buckets。Prefill CUDA Graph 已禁用，因此不另外配置 prefill buckets。

静态 contract 位于 [`configs/matrix/qwen35_dapo_thinking_replayssm.yaml`](../configs/matrix/qwen35_dapo_thinking_replayssm.yaml)。它引用两个 verifier 配置、一个 drafter 配置、现有 DAPO client 和 observability 配置；campaign manifest 会保存这些 YAML 的 SHA-256 与完整快照，所以之后源文件漂移不会被悄悄混入已物化 campaign。

## Target 为什么不等 drafter prefill

per-request 的启动序列是：

```text
verifier 接收请求
  → 立即执行 target prefill，同时异步发出 DraftSync
  → GPU tail 暂时无可消费 proposal：本轮 selected_len=0，target-only
  → drafter 完成自己的 prefill 并开始 decode
  → draft token 经 ZMQ 到 verifier daemon，异步 H2D 到 GPU tail
  → 之后的 verify 轮次在 current lifecycle/base 匹配时消费 proposal
```

这个设计把 drafter prefill 从 target critical path 上拿掉。运维层仍要确认
verifier/drafter 两个独立 server 已就绪；但“server 就绪”不等于“每个
target request 必须等 drafter prefill 完成”。

因此，一个很短的冷启动 case 可能在首批 draft 到达前就完成，最终
`spec_num_proposed_drafts=0`。这不是协议错误，也不应用等待 gate 去“修好”；
它表示本 case 没有实际锻炼 decoupled proposal 路径，所以 summary 会保留
run 并标记 `review: no_draft_proposals`。对 1K–32K 的正式长输出矩阵，
`proposed=0` 仍是需要检查 drafter 启动、DraftSync、tail landing 和供给时序的
usefulness diagnostic，但不会被写成 target correctness failure。

## v5 在哪些 production 边界上区别于旧 campaign

v5 的 model/workload/config contract 没有改变，区别是正式 source 包含此前三项
lifecycle/transport hardening，以及 terminal scatter、static topology 和 request-slot
accounting 修复：

- **chunked close/abort ownership**：drafter 收到 close 时，在释放 checkpoint/KV
  前先清除 `chunked_req` 和 `_pending_chunked_abort_req`；verifier 的 pending
  chunked abort 也在 generic scheduler release 前先通知 decoupled component 关闭
  mirror，防止已释放 request 被下一个 chunk 重新 stash/调度；
- **`pause_generation(retract)` component hook**：generic `retract_all()` 前，
  verifier component 对每个唯一 request 关闭 mirror；drafter component 把 sleeping
  rows 合并进 retract set，每个 request 只做一次 rollback/checkpoint reset，再交给
  scheduler 统一释放和 requeue；
- **C++ ZMQ interruptible/bounded queues**：verifier control 与 drafter tail 都使用
  nonblocking send 和 1 ms poll slice 检查 shutdown，peer 不存在或崩溃时 close
  可中断；队列上限为 8,192 frames/512 MiB，超限 fail fast，并保持
  queue-front retry 和 frame order。1 ms 只是 shutdown cancellation granularity，不是
  delivery deadline，也不是 snapshot timeout；
- **terminal scatter**：用 GPU `scatter_` 替代 Python scalar advanced indexing，
  消除 overlap hot path 中的隐式 `cudaStreamSynchronize`；
- **static top-k=1 topology**：按 request-pool capacity 预分配线性 tree topology，
  每轮只 slice；
- **request-slot accounting**：continued chunked/session request 已拥有
  `req_pool_idx` 时不重复消耗新 slot budget，BS64 不再退化为 63 running + 1 waiting。

对应 focused tests 覆盖 mid-chunk close ordering、chunked abort delegation、pause retract
去重与 sleeping-row reset，以及 no-peer close、queue saturation、late-connect order 和
peer-crash close。这些修复改变了正式 production source 边界，所以即使
campaign contract SHA 不变，但 source hash 已改变，因此 v2/v3/v4 的 sealed run
都不能代替 v5 重跑。

## 为什么有三层 gate

矩阵把“能启动”“测量有效”和“结果看起来正常”分开：

1. **Campaign 前置 gate**：真实 BS8 inbox/segment correctness probe。未通过时，不能注册 attempt。
2. **每轮 boot hard gate**：从两个独立 server log 中读取实际 `max_total_num_tokens`、Mamba pool size 和 C++ data-plane selection。Verifier/drafter token capacity 都必须至少为 2,106,640；Mamba pool 分别至少为 320/512。
3. **Sealed result hard gate**：sealed audit、checksum、四份 resolved config、batch
   cardinality、固定输出 token 数和 `spec_verify_ct>0` 都必须通过。
   `proposed=0` 只是 diagnostic，不是 hard failure；这是 no-wait startup
   语义必须保留的边界。

2,106,640 的容量需求来自已 materialize 的 first-64 DAPO thinking workload：prompt length 为 min=94、max=217、sum=9,232；最重 case 的需求为：

```text
9,232 prompt tokens + 64 × 32,768 output tokens + 64 × 4 verify reserve
= 2,106,640 tokens
```

机器可读 preflight 是 [`qwen35_dapo_thinking_bs64_out32768.inspect.json`](../configs/matrix/qwen35_dapo_thinking_bs64_out32768.inspect.json)，由现有 `inspect_workload.py` 以 `--batch-size 64 --output-len 32768` 生成；contract 固定其 SHA-256 `155dc5489c7e873620f8f69a27bcea40c1d14d001c78207a413fdf8aaf62a6f4`，并重新计算上述 prompt sum 和容量公式。该文件只 materialize workload，不发送请求。

存在实际 proposals 时，`spec_accept_rate>=0.50`、`spec_draft_occupancy_rate>=0.25` 和 overlap 相对 non-overlap 的 accept-rate drop 不超过 0.10 是 **诊断阈值**，不是删除数据的理由。触发阈值的 sealed run 仍保留，但 summary 标为 `review`。其中 accept rate 衡量“实际提出 token 的正确率”，occupancy 衡量“实际供给占 K-step nominal capacity 的比例”，两者不能混报；没有 proposal 时 accept rate 为 N/A，而不是 0。

## 静态校验与 materialization

从目标 worktree 根目录执行：

```bash
MATRIX_CONFIG=benchmark/decoupled_spec/configs/matrix/qwen35_dapo_thinking_replayssm.yaml

PYTHONPATH=python python benchmark/decoupled_spec/matrix/campaign.py validate \
  --config "$MATRIX_CONFIG"

CAMPAIGN_DIR=benchmark/decoupled_spec/results/campaigns/qwen35-dapo-replayssm-scaling-postslot-v5
PYTHONPATH=python python benchmark/decoupled_spec/matrix/campaign.py materialize \
  --config "$MATRIX_CONFIG" \
  --campaign-dir "$CAMPAIGN_DIR"
```

`materialize` 只生成 `campaign_manifest.json` 和可恢复的 `ledger.json`，
不加载模型、不申请 GPU，也不启动任何 role。相同 contract 重复调用是
幂等的；不同 contract 不能覆盖已有 campaign。特别是，v5 已把
`min_spec_num_proposed_drafts` 固定为 0，并用
`diagnostic.require_nonzero_spec_proposals=true` 保留 usefulness 检查；不得把旧
`proposed>0` hard-gate manifest 覆盖到 v5 目录，也不得把旧 campaign 的 attempt
记录或 RUN_DIR 复制到 v5 ledger。

该前置 gate 已由以下 sealed BS8 probe 打开：

```text
benchmark/decoupled_spec/results/
20260821T101836-prereq-inbox-segment-bs8-out256-thinking-overlap-cpp-replayssm-62d6f2a591e9
```

它完成 8/8 request、2,048 output tokens，`spec_verify_ct=1067`、actual
acceptance=`0.8242`、occupancy=`0.3714`，并通过 pre-seal/sealed audit 与
checksum 复核。如需重新物化新 contract，仍必须用可追溯 evidence 显式
打开 gate：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/matrix/campaign.py set-prerequisite \
  --campaign-dir "$CAMPAIGN_DIR" \
  --gate-id draft_inbox_segment_bs8_correctness \
  --state passed \
  --evidence /absolute/path/to/the/bs8/probe/artifact \
  --note "real verifier/drafter BS8 correctness probe passed"
```

不能用口头结论或一个不存在的路径代替 evidence；执行 Agent 应先检查该产物本身的配置、输出正常性和进程最终状态。

## 每个 case 仍由四个原子角色完成

查看一个 case 的 exact contract 和命令 argv：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/matrix/campaign.py show-case \
  --campaign-dir "$CAMPAIGN_DIR" \
  --case-id overlap-bs8-out1k
```

然后按根目录 benchmark 的标准生命周期执行：

```text
preflight
  → artifacts.py init 创建唯一 RUN_DIR
  → register-attempt
  → verifier_server.py 与 drafter_server.py 分别启动
  → wait_for_roles.py
  → transition servers_ready（自动检查 boot log hard gate）
  → collector.py 单独启动并取得双端 baseline
  → client.py 提交一次 tokenized streaming batch
  → 双端 trailing samples
  → 只停止本轮持有的 collector/verifier/drafter
  → 四个单轮 plot/report 脚本
  → pre-seal audit
  → seal + sealed audit + checksum
  → transition verified（再次执行 boot、exact-config 和 sealed gates）
```

这里没有 `run.py`、combined server 或隐藏 launcher。Campaign manifest 中的 argv 只是把 case 轴绑定到现有独立入口；进程 ownership、readiness、collector 和 client 仍由 Agent 分别管理。

创建 RUN_DIR 时必须使用 case 中的精确 `run_name`。例如：

```bash
RUN_NAME=qwen35-27b-tp4-draft-0.8b-tp1-k3-f1-bs8-dapo-thinking-out1024-overlap-cpp-gpu-tail-replayssm
RUN_DIR="$(PYTHONPATH=python python benchmark/decoupled_spec/common/artifacts.py init \
  --output-root benchmark/decoupled_spec/results \
  --name "$RUN_NAME")"

PYTHONPATH=python python benchmark/decoupled_spec/matrix/campaign.py register-attempt \
  --campaign-dir "$CAMPAIGN_DIR" \
  --case-id overlap-bs8-out1k \
  --run-dir "$RUN_DIR"
```

每完成一个阶段，用 `campaign.py transition` 前移状态。`servers_ready` 和 `verified` 不是纯记账：它们会执行对应 hard gate。Client 的精确覆盖参数始终是：

```bash
--batch-size <BS> --output-len <OUTPUT_LEN>
```

这仍然表示一次 `/generate` 中包含 BS 个 tokenized input，而不是 BS 个并发 HTTP 请求。

## 失败、恢复和容量剪枝

失败 attempt 必须停止自己拥有的进程，并记录最早有用错误：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/matrix/campaign.py transition \
  --campaign-dir "$CAMPAIGN_DIR" \
  --case-id overlap-bs8-out1k \
  --state incomplete \
  --failed-stage client \
  --error "the first useful error"
```

该 RUN_DIR 保持 unsealed、不可覆盖、不可删除。修复后使用新的唯一 RUN_DIR 注册下一次 attempt，ledger 会保留完整历史。

只有确认是 capacity failure 时才做 dominance 剪枝：在相同 mode 下，同时满足 `batch_size>=failed_bs` 且 `output_len>=failed_output_len` 的 case 可以标为 `blocked`。普通 correctness、transport 或 process failure 不能据此跳过其它 case。阻塞状态和证据 case 都写入 ledger，外部状态变化后可 `unblock-case` 恢复。

推荐按 `nonoverlap → overlap`、BS 递增、output length 递增执行。这样同一 `(BS, output)` 先获得 control，较小 capacity 点也能尽早发现问题。

## 只从 sealed runs 生成跨 case 报告

每轮仍先使用现有 `plot/*.py` 和 artifact audit 生成、验收单轮报告。跨 case 汇总只读 ledger 中 `state=verified` 的 sealed run：

```bash
PYTHONPATH=python python benchmark/decoupled_spec/matrix/summarize.py \
  --campaign-dir "$CAMPAIGN_DIR"

# 最终交付时要求 32/32；不完整会直接失败
PYTHONPATH=python python benchmark/decoupled_spec/matrix/summarize.py \
  --campaign-dir "$CAMPAIGN_DIR" \
  --require-complete
```

输出位于 `$CAMPAIGN_DIR/summary/`：

- `matrix_summary.json`：完整 case rows、status counts、阈值和 source provenance；
- `matrix_summary.csv`：便于筛选的逐 case 指标；
- `matrix_report.md`：结论优先的结果表与 paired overlap 对照；
- `performance_report.md`：perf audit、修复机制、32-case 结论与证据边界；
- `experiment_progress_report_zh.md`：面向人阅读的中文进展、快慢原因与下一步；
- `plots/throughput_by_batch.{svg,png}`：四个输出长度下的 BS scaling；
- `plots/paired_overlap_effects.{svg,png}`：throughput、acceptance、occupancy 和
  output/verify 的 paired overlap 变化；
- `manifest.json`：campaign manifest、ledger 和每个 sealed RUN_DIR 的 checksum/source hash。

汇总会逐请求比较 fixed-seed overlap/non-overlap `output_ids`，并保留每个请求
的首个 token diff。这只是 informational evidence：普通 BS>1 性能配置的
dynamic batching、asynchronous proposal grouping 和 reduction order 不受 token-exact
contract 约束，所以 mismatch 本身不会把 case 变成 `review`，更不会被伪装成
已定位的 overlap correctness failure。`--require-complete` 仍严格要求 32/32
sealed + verified case。

### “生成内容正常”如何做初步检查

性能矩阵的初步 content sanity 不判定数学答案是否正确，而是检查数据链
没有损坏或串请求：

- `sampled_requests.jsonl`、raw batch response 和 `responses.jsonl` 的
  cardinality、index、request ID、DAPO row index、prompt length 与 generated
  text 绑定必须一致；
- `output_ids` 和 metadata completion count 必须达到 requested fixed length，
  finish reason 与 fixed-length run 一致；
- decoded text 必须非空、可 UTF-8 表示、不含 NUL，Unicode replacement
  character `U+FFFD` 比例不超过 1%；
- 每请求保留 input/output/text SHA-256 和 head/tail snippet，既方便人工看
  是否是正常 thinking/text，也能查找重复、串行或后处理污染。

汇总 row 使用 `content_sanity_status=pass|review`、
`content_sanity_issue_count`、`content_sanity_healthy_request_count`、
`content_text_sha256`、`content_input_binding_sha256`、first/last request 的四个
head/tail 字段，以及 nested `content_sanity` 保留可追溯细节。任一绑定或
文本检查失败都会添加 top-level `content_sanity_failed`；跨 run 的同
request index 如果 input binding 不一致，per-request issue 会记录
`input_binding_mismatch_across_runs`。
这是“内容形态和请求归属正常”的初步证据，不是 task-level accuracy oracle。

## ReplaySSM 和 token-exact 的证据边界

若长输出出现 token-level 分歧，使用 `configs/verifier/controls/` 下的
matched recurrent verifier 配置复跑同一个 case。它们只关闭
`enable_linear_replayssm_spec` 并显式保持 `mamba_ssm_dtype=float32`；
topology、extra_buffer、K/F、seed、CUDA Graph、token/Mamba pool 均与正式矩阵
相同。这个 control 用于判断 ReplaySSM 是否是某个 divergence signature 的
必要条件，不进入 32-case 性能汇总。

当前 debug control 还证明，即使 verifier/drafter 都开启
`enable_deterministic_inference`，decoupled 的 ZMQ/daemon/tail arrival 与 proposal grouping
仍然不在该 flag 的确定性 contract 内。因此，它不是本 BS>1 矩阵的
必选开关，也不能把“开了该 flag 仍未 repeat exact”直接当作 verifier overlap
bug。如要建立真正的 token-exact hard oracle，还必须先固定 proposal trace/grouping
并证明同配置 repeat exact；这是独立 control lane，不是当前性能 campaign
的通过条件。

### 已保留的 diagnostic reports（不是正式矩阵结果）

- [原始 BS8 ReplaySSM 输出分叉回放包](../results/campaigns/qwen35-dapo-replayssm-scaling/debug/replayssm_output_divergence_report.md)：
  保留旧未固定 seed 四个 sealed run 的 inputs、first diff 和边界；
- [seed42 overlap 1K/4K prefix control](../results/campaigns/qwen35-dapo-replayssm-scaling/debug/seed42_overlap_1k_4k_prefix_report.md)：
  该次 8/8 前缀 exact，但只是一组 matched-seed pair；
- [seed42 4K non-overlap/overlap 对照](../results/campaigns/qwen35-dapo-replayssm-scaling/debug/seed42_4k_nonoverlap_vs_overlap_report.md)：
  分叉不贴 peer finish 或当轮 ReplaySSM track boundary，但未定位因果；
- [seed42 overlap recurrent repeats](../results/campaigns/qwen35-dapo-replayssm-scaling/debug/seed42_overlap_recurrent_repeat_report.md)：
  关闭 ReplaySSM 后仍复现同类 signature，所以 ReplaySSM 不是必要条件；
- [deterministic decoupled repeats](../results/campaigns/qwen35-dapo-replayssm-scaling/debug/deterministic_decoupled_repeat_report.md)：
  overlap 和 non-overlap 都可在相同记录配置下因 proposal grouping 不同而未
  token-exact，排除“verifier overlap scheduler 是必要条件”。

这些报告的 debug run 多位于 `results/incomplete/`，未 seal、未进入性能汇总；
报告中的 hash 和配套 JSON 用于追溯读取过的 artifact，不把 diagnostic
强化为 correctness 或性能结论。

汇总脚本不会读取 incomplete RUN_DIR、不会发送流量，也不会向 sealed RUN_DIR 写入任何文件。
