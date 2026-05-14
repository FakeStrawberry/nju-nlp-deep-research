# vLLM 版教学 Agent

这个目录是面向课堂教学的 **精简版实现**，默认假设：

- 模型服务已经由 `vLLM` 启动
- 本地只需要提供：
  - `BrowseComp-Plus` 的 BM25 检索工具
  - 一个最小的 RAG 示例
  - 一个可供参考的完整 agent-loop notebook

## 教学分层

### 学生版

- notebook: [agent_vllm.ipynb](/mnt/d/桌面/南大nlp/老师/nlp助教/agent_vllm.ipynb)
- 只展示：
  - 单次 `search`
  - 将检索结果拼进 prompt
  - 调用 vLLM 输出最终答案
- notebook: [agent_vllm_weather.ipynb](/mnt/d/桌面/南大nlp/老师/nlp助教/agent_vllm_weather.ipynb)
  - 用本地模拟 `get_weather` 工具测试 vLLM + agent 的工具调用链路
  - 不依赖检索语料和外部 API

不提供完整 agent loop，让学生自己实现多轮工具调用。

### 标准答案版

- notebook: [agent_vllm_answer.ipynb](/mnt/d/桌面/南大nlp/老师/nlp助教/agent_vllm_answer.ipynb)
- 展示：
  - OpenAI-compatible `tools`
  - 多轮 agent loop
  - `search` / `get_document` 两个更细粒度工具
  - 轨迹记录

## 目录说明

- [browsecomp_searcher.py](/mnt/d/桌面/南大nlp/老师/nlp助教/agent/browsecomp_searcher.py)
  - 本地 SQLite FTS5 BM25 检索实现
- [build_bm25_index.py](/mnt/d/桌面/南大nlp/老师/nlp助教/agent/build_bm25_index.py)
  - 服务器上预构建索引
- [vllm_client.py](/mnt/d/桌面/南大nlp/老师/nlp助教/agent/vllm_client.py)
  - vLLM OpenAI-compatible client
- [tools.py](/mnt/d/桌面/南大nlp/老师/nlp助教/agent/tools.py)
  - RAG 检索与 agent tools 定义
- [dataset_utils.py](/mnt/d/桌面/南大nlp/老师/nlp助教/agent/dataset_utils.py)
  - 读取 `hard50` 测试集等辅助函数

## 服务器上的一次性准备

先构建 BM25 索引：

```bash
python -m agent.build_bm25_index \
  --corpus-path ./browsecomp-plus-corpus \
  --index-path ./indexes/browsecomp_plus_bm25.sqlite \
  --overwrite
```

之后 notebook 只需要复用 `index_path`。

## 多轮 Deep Research Agent 脚本

基础任务实现放在 [deep_research_agent.py](deep_research_agent.py)。

它做的事情：

- 用 `search` / `open_doc` / `find_in_doc` 执行本地检索与文档检查
- 让模型在多轮 loop 中决定是否继续搜索或最终回答
- 用最大轮数与“无新信息”作为停止条件
- 保留完整 `messages` 轨迹，并额外记录 `state_summary`
- 直接写出评测脚本可读取的 JSONL

示例：

```bash
python -m agent.deep_research_agent \
  --dataset browsecomp_plus_hard50.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --output runs/submission.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --top-k 8 \
  --max-rounds 6
```

调试时可以先只跑 3 条：

```bash
python -m agent.deep_research_agent \
  --dataset browsecomp_plus_hard50.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --output runs/debug_submission.jsonl \
  --model qwen_auto \
  --limit 3
```

## 依赖

```bash
pip install -r agent/requirements.txt
```

最小依赖：

- `pyarrow`
- `python-dotenv`

## vLLM 说明

这一版不再自己实现模型服务层，直接依赖已经启动好的 vLLM OpenAI-compatible endpoint。

根据 vLLM 官方文档：

- 自动工具调用需要 `--enable-auto-tool-choice`
- 还需要为对应模型设置 `--tool-call-parser`
- 推理模型如果需要 reasoning 解析，还要匹配 `--reasoning-parser`
- 如果模型输出是自定义 tool 格式，还要通过 `--tool-parser-plugin` 注册本地 parser

参考文档：

- [Tool Calling - vLLM](https://docs.vllm.ai/en/stable/features/tool_calling/)
- [Qwen3ReasoningParser - vLLM](https://docs.vllm.ai/en/latest/api/vllm/reasoning/qwen3_reasoning_parser/)

## Pangu / DeepDiver 兼容

`agent/pangu_tool_parser.py` 提供了一个给 Pangu / DeepDiver 使用的 vLLM tool parser，
默认识别：

- `[unused11] ... [unused12]` 中的 JSON tool call
- `[unused16] ... [unused17]` 中的 reasoning
- markdown 代码块中的 JSON tool call
- `tool` / `name` / `function` / `tool_name` 等常见字段写法

启动示例：

```bash
vllm serve ./openPangu-Embedded-7B-DeepDiver \
  --served-model-name pangu_auto \
  --enable-auto-tool-choice \
  --tool-parser-plugin agent/pangu_tool_parser.py \
  --tool-call-parser pangu_deepdiver \
  --chat-template agent/pangu_chat_template.jinja \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

教学场景下不再优先推荐 vLLM 内置 `pangu` parser，因为 DeepDiver 有时会输出
说明文字加 markdown JSON，而不是单一严格格式。自定义 parser 的作用就是在服务端
把这些差异抹平，尽量稳定地产生 OpenAI `tool_calls`。

如果你的 tokenizer 里已经带了可用的 tool template，也可以把 `--chat-template`
换成模型自带的模板设置。

## Qwen3-8B 兼容

如果你们要直接起 Qwen3-8B 服务，`agent/` 侧可以直接走 vLLM 内置的 Qwen3 parser：

```bash
vllm serve ./Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```
