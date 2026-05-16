# Deep Research Agent 版本更新日志

本文记录本项目从 baseline 到 v7 的主要改动、设计意图、实验效果与阶段性反思。

## 评估口径说明

- 数据集：`browsecomp_plus_hard50.jsonl`，共 50 题。
- 检索来源：本地 `browsecomp-plus-corpus/` 构建的 BM25 索引。
- 评估脚本：`agent/eval.py`，使用 `qwen_auto` 作为自动 judge。
- 自动 judge 存在噪声，尤其是模型输出 `Insufficient evidence` 时，可能被误判为正确。因此本文同时记录自动评估结果和人工复核观察。
- 所有版本都不使用外部搜索服务、不替换 BM25 检索器、不把标准答案写入 prompt 或代码。

## 总览

| 版本 | 输出文件 | 自动评估结果 | 主要变化 | 核心意图 |
| --- | --- | ---: | --- | --- |
| v1 | `runs/submission.jsonl` | 1/50 = 2% | 初版可运行流程 | 跑通端到端生成、提交和评估 |
| v2 | `runs/submission_v2.jsonl` | 4/50 = 8% | 增强检索轮次与工具调用 | 解决模型过早作答、检索不足 |
| v3 | `runs/submission_v3.jsonl` | 6/50 = 12% | 加强多轮 research loop、上下文状态与自动打开文档 | 提高证据覆盖率与多跳探索能力 |
| v3 eval-1024 | 同 v3 submission | 10/50 = 20% | 只调整评估 max tokens | 验证自动评估对 judge 输出长度敏感 |
| v4 | `runs/submission_v4.jsonl` | 9/50 = 18% | query-aware snippet | 让检索结果摘要更贴近当前 query |
| v5 | `runs/submission_v5.jsonl` | 11/50 = 22% | 最终答案验证检索 | 修正缩写、候选答案不稳和证据不足问题 |
| v6 | `runs/submission_v6.jsonl` | 8/50 = 16% | BM25-aware query planner、自适应补检索、候选验证清洗 | 提高关键证据召回，降低无效候选反查 |
| v7 | `runs/submission_v7.jsonl` | 待评估 | LLM-first query、gated fallback、候选仲裁 | 保留 v5 稳定性，吸收 v6 的受控召回收益 |

## v1：端到端 baseline

### 改动

- 跑通从数据读取、调用模型、生成 `submission.jsonl`、再用 `agent/eval.py` 评估的完整流程。
- 初步接入本地 BM25 检索工具。
- 产出统一 JSONL 格式，方便后续版本对比。

### 意图

v1 的主要目标不是追求高准确率，而是确认工程链路可用：模型服务、数据集、检索索引、输出格式和评估脚本都能连起来。

### 效果

自动评估：

- 正确：1/50
- 准确率：2%
- 平均工具调用：1.3 次/题
- 平均检索文档：0.06 篇/题

### 反思

v1 的最大问题是模型很容易在证据不足时直接给最终答案，实际检索量极少。说明只把工具暴露给模型不够，还需要显式设计 agent loop、最少检索要求、停止条件和上下文管理。

## v2：强化检索与工具调用

### 改动

- 提高每题的检索和工具调用预算。
- 让 agent 更积极地使用 `search`、`open_doc` 等本地工具。
- 减少模型在没有足够证据时过早 final answer 的情况。

### 意图

v2 针对 v1 的核心短板：不是模型完全不会回答，而是它没有拿到足够多的候选证据。该版本优先扩大证据覆盖面，让模型先“查得更多”。

### 效果

自动评估：

- 正确：4/50
- 准确率：8%
- 平均工具调用：8.0 次/题
- 平均检索文档：36.0 篇/题

### 反思

v2 相比 v1 明显提升，说明增加检索是有效的。但同时也暴露出新问题：检索结果多了以后，模型不一定能稳定筛出关键证据。单纯增加 top-k 或工具调用次数，会带来更多噪声和上下文压力。

## v3：更完整的 research loop

### 改动

- 加强多轮 agent loop，围绕“搜索、打开文档、查找文档内关键词、再总结”的流程推进。
- 加入更明确的状态管理，记录已搜索 query、已见 docid、证据摘要和工具调用次数。
- 增加自动打开 top 文档的机制，避免模型只看 search snippet 就下结论。
- 使用最大轮数和“连续无新信息”作为停止条件，避免无限检索。

### 意图

v3 的目标是从“多查一点”升级为“有结构地查”。BrowseComp-Plus 的问题往往需要多跳线索，只靠单次搜索结果不稳定，因此需要 agent 维护中间状态，并基于已有证据决定下一步。

### 效果

自动评估：

- 正确：6/50
- 准确率：12%
- 平均工具调用：14.82 次/题
- 平均检索文档：69.16 篇/题

额外观察：

- 使用同一个 `submission_v3.jsonl`，把评估脚本的 `--max-tokens` 提高到 1024 后，自动评估变为 10/50。
- 这不代表 agent 本身变强了，而是说明自动 judge 对输出长度和判断格式比较敏感。

### 反思

v3 证明多轮检索和状态管理能继续提升结果，但也说明“拉满检索次数”不是万能的。检索越多，噪声也越多；如果 snippet 不够贴近问题，模型会被大量无关文本稀释注意力。

## v4：query-aware snippet

### 改动

- 在 `agent/tools.py` 中加入 query-aware snippet。
- `search` 仍然使用原本的 BM25 排名，不改变检索器、不替换索引、不过滤 query。
- 改动只发生在结果展示层：从文档中截取更靠近 query 命中词的位置，而不是固定返回文档开头。

### 意图

v3 的失败分析显示，很多时候相关文档已经被检索到，但返回给模型的 snippet 没有覆盖真正有用的句子。v4 的目标是让同样的 BM25 结果提供更有信息密度的摘要，提升模型判断候选证据的能力。

### 效果

自动评估：

- 正确：9/50
- 准确率：18%
- 平均工具调用：14.82 次/题
- 平均检索文档：69.16 篇/题

人工复核观察：

- 自动评估里有明显误判，尤其是预测为 `Insufficient evidence` 但被 judge 判正确的样本。
- 较保守地看，v4 真实可靠正确数约为 7/50。
- v4 相比 v3 确实新增了一些有效命中，例如更容易从检索片段里看到关键数字或实体。
- 但也出现个别退化，例如模型把更完整的实体名压缩成 ticker 或简称。

### 反思

query-aware snippet 是合规且有效的基础优化，因为它不改变检索范围和排名，只改善证据呈现方式。但它不能解决最终答案规范化问题：模型可能看到正确证据，却输出缩写、简称、过长解释或不稳定候选。

## v5：最终答案验证检索

### 改动

- 在 `agent/deep_research_agent.py` 中加入最终答案验证阶段。
- agent 先按 v4 流程完成检索和推理，得到候选 `Exact Answer`。
- 如果开启验证，系统会基于候选答案和已有 query 追加最多 2 次验证检索。
- 每次验证检索会自动打开 top1 文档。
- 最后让模型只基于已有证据和验证结果重新确认最终答案。
- 新增参数：
  - `--verification-top-k`
  - `--verification-open-top-n`
  - `--no-verify-final-answer`

### 意图

v5 针对 v4 的两个典型问题：

1. 候选答案已经接近正确，但形式不符合标准答案，例如 ticker、缩写、简称。
2. 模型在证据不稳定时直接 final，缺少最后一步交叉验证。

最终验证不是重新做一套检索系统，而是在提交前对候选答案做 evidence check 和 normalization。

### 预期效果

v5 预计能改善：

- `SPRO` 这类 ticker 输出，若证据支持，应规范成完整公司名。
- 候选答案与文档证据不一致的情况。
- 模型过早输出 `Insufficient evidence` 的情况。

v5 可能带来的风险：

- 工具调用次数会增加，运行时间会上升。
- 如果验证检索结果噪声较大，模型可能过度修正原本正确的答案。
- 如果候选答案本身很差，基于候选的验证 query 也可能继续偏离。

### 实测效果

自动评估：

- 正确：11/50
- 准确率：22%
- 平均工具调用：12.0 次/题
- 平均检索文档：47.94 篇/题

与 v4 对比：

- 自动评估从 9/50 提升到 11/50。
- 工具调用从 14.82 次/题下降到 12.0 次/题，检索文档从 69.16 篇/题下降到 47.94 篇/题。
- 新增 2 个自动判正确样本，没有丢失 v4 自动判正确样本。

人工复核观察：

- 其中 1 个新增样本属于真实收益，v5 将候选答案规范成了完整标题。
- 仍有若干自动 judge 噪声：预测为 `Insufficient evidence` 但被判正确。
- 在 v5 的 39 个自动判错样本中，只有 6 个样本的标准答案字面出现在轨迹中，说明主要瓶颈仍是关键证据召回，而不是最终答案格式。
- v5 验证阶段有 9 题改变了候选答案，但部分候选来自不规范模型输出，如 `Wait` 或长段推理文本，这会污染候选反查 query。

### 反思

v5 的方向比继续盲目增加检索次数更成熟。前几版结果说明，单纯扩大检索预算收益递减，后期瓶颈更多在“候选答案是否被证据支持”和“答案形式是否规范”。不过 v5 的实测也说明，最终验证只能解决后处理问题，不能替代更好的 query 规划；如果关键文档没有被召回，验证阶段无法凭空修复。

## v6：BM25-aware query planner 与自适应补检索

### 改动

- 重写初始 query 规划顺序，优先生成更适合当前 SQLite FTS5/BM25 的短 query：
  - 引号短语 query
  - 大写实体短语 query
  - 编号、代码、年份 query
  - 罕见词 query
  - LLM query plan 压缩版
  - 原始问题兜底
- 新增自适应补检索机制：
  - 当一轮工具调用没有新文档或新证据时，自动追加少量未尝试过的 deterministic fallback query。
  - 当模型准备以 `Insufficient evidence` 或不规范答案收尾时，若预算允许，先追加 fallback query 再继续推理。
- 新增参数：
  - `--adaptive-query-count`
  - `--max-adaptive-searches`
- 清理最终验证候选：
  - 只有格式正常、短且非 `Insufficient evidence` 的 `Exact Answer` 会用于候选反查。
  - 如果模型输出残片或长段推理文本，不再把它当作 candidate query。
- 更新最终回答和验证 prompt，要求在有具体候选证据时优先给出低置信度具体答案，而不是过早输出 `Insufficient evidence`。

### 意图

v6 直接针对 v5 暴露出的主要问题：多数错误不是因为最终答案格式，而是因为关键证据没有被召回。结合当前检索器实现，query 会被拆成去重 token 并通过 `OR` 匹配，因此长 query 里的普通词容易制造噪声。v6 的目标是让 query 更短、更实体化、更适合 BM25 稀疏匹配。

### 实测效果

自动评估：

- 正确：8/50
- 准确率：16%
- 平均工具调用：13.44 次/题
- 平均检索文档：54.3 篇/题

与 v5 对比：

- 自动评估从 11/50 下降到 8/50。
- 工具调用从 12.0 次/题上升到 13.44 次/题，检索文档从 47.94 篇/题上升到 54.3 篇/题。
- v6 新增 5 个自动判正确样本，但丢失 8 个 v5 自动判正确样本。
- v5 与 v6 同时正确的样本只有 3 个，说明两版存在互补性，但 v6 的选择机制不稳定。

人工复核观察：

- v6 的 BM25-aware query planner 能找到一些 v5 没找到的关键线索，说明 query 层优化方向有价值。
- 但 deterministic 短 query 太靠前，容易引入同名实体和宽泛噪声，例如半截实体、过泛机构名或年月类 query。
- adaptive search 默认最多 2 次，带来了更多文档，但没有稳定提升最终答案。
- verifier 在没有有效候选时仍可能输出 `Insufficient evidence` 或格式残片。

### 反思

v6 没有替换检索器，也没有引入外部知识；它只改变 query 生成和停滞处理策略。实测表明，query 层优化不能只追求更广召回，还必须控制 query 质量和进入上下文的证据噪声。v7 因此应回到 v5 的稳定主线，同时把 v6 的 deterministic query 收缩为受控补充。

## v7：LLM-first query 与候选仲裁

### 改动

- 恢复 LLM-first bootstrap：
  - 初始检索重新优先使用模型规划出的 query。
  - BM25-aware deterministic query 不再抢占第一批主搜索，只在通过质量门控后作为补充。
- 加入 deterministic query 质量门控：
  - 过滤半截实体、单词 query、泛机构 query、纯年份/泛词 query。
  - 保留明确编号、代码、标题、人名、机构全称和含稀有词的组合 query。
- 收缩 adaptive search：
  - 默认 `--max-adaptive-searches` 从 2 降到 1。
  - fallback query 必须通过质量门控。
- 将最终验证改为候选仲裁：
  - 从主回答和历史 assistant 消息中收集格式正常的候选答案。
  - verifier 接收候选列表，而不是单个伪候选。
  - 如果已有具体候选，verifier 不能轻易用 `Insufficient evidence` 覆盖它。
  - 若 verifier 输出的具体答案没有被证据摘要支持，而原候选有证据支持，则保留原候选。

### 意图

v7 的目标不是继续增加搜索量，而是降低 v6 的噪声副作用。它保留 v6 对 BM25 query 特性的利用，但把 deterministic query 放到更保守的位置；同时把 v5 的最终验证升级成候选仲裁，减少单一候选错误或无效候选导致的后处理退化。

### 当前效果

截至本文记录时，v7 代码已完成并通过静态检查，完整 `hard50` 评估尚未运行。因此 v7 的最终效果需要以 `runs/eval_results_v7.jsonl` 为准。

推荐运行：

```bash
python -m agent.deep_research_agent \
  --dataset browsecomp_plus_hard50.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --output runs/submission_v7.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --top-k 8 \
  --max-rounds 6 \
  --max-tokens 1024 \
  --bootstrap-query-count 5 \
  --auto-open-top-n 1 \
  --min-tool-calls 3 \
  --verification-top-k 5 \
  --verification-open-top-n 1 \
  --adaptive-query-count 1 \
  --max-adaptive-searches 1
```

评估：

```bash
python -m agent.eval \
  --submission runs/submission_v7.jsonl \
  --dataset browsecomp_plus_hard50.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/eval_results_v7.jsonl \
  --max-tokens 1024
```

### 反思

v7 是一次收缩型改动：不扩大预算，而是控制 query 质量和最终答案替换条件。如果 v7 能恢复 v5 的稳定正确样本，同时保留部分 v6 新增命中，说明“LLM-first + gated fallback”比“deterministic-first”更适合当前 BM25 工具。

## 阶段性总体反思

1. 端到端链路比单点 prompt 更重要。v1 到 v3 的提升主要来自 agent loop、工具调用和状态管理。
2. 检索量不是越大越好。v2、v3 说明更多文档能提高覆盖率，但也会引入噪声，需要更好的证据选择。
3. snippet 质量很关键。v4 的 query-aware snippet 证明，在不改变 BM25 的前提下，仅改善文档片段呈现就能提升模型可用证据。
4. 自动评估只能作为参考。v3 eval-1024 和 v4 都显示 judge 会产生误判，因此需要结合人工复核分析。
5. 后续优化应优先做通用机制，不应针对 hard50 的标准答案或具体题目做规则。
6. v5 到 v6 的分析表明，关键证据召回仍是主瓶颈，但召回必须受质量门控约束。

## v8 可继续尝试的方向

### 方向一：证据级 rerank

仍然先用 BM25 检索 top-k，但在返回给模型前，对 top-k 内部结果做轻量重排。例如优先展示同时包含问题关键实体、候选实体、数字、日期或标题词的片段。

这个方向不替换 BM25，只是在已有结果内部做证据排序，合规风险较低。

### 方向二：答案类型约束

根据问题自动判断答案类型，例如人名、机构、作品名、数字、日期或地点。最终答案阶段要求输出与类型匹配的短答案，减少句子型答案、缩写答案和 `Insufficient evidence` 误用。

### 方向三：多候选比较

最终阶段不只验证一个 candidate，而是让模型列出 2 到 3 个候选答案，再逐个检查证据强度，最后选择证据最直接的一项。

### 方向四：失败状态触发补检索

v6/v7 已经实现轻量级停滞补检索。后续可以继续细化触发条件：当最终答案没有出现在任何 opened doc、证据引用为空、或候选答案只在 snippet 中弱出现时，自动触发更有针对性的补检索。这个机制可以只看当前轨迹，不需要标准答案。

### 方向五：评估稳定性改进

对 `agent/eval.py` 做更稳健的输出解析和更长 `max_tokens`，减少自动 judge 因格式截断或解释不足产生的误判。这个方向只影响分析和调试，不改变提交答案。
