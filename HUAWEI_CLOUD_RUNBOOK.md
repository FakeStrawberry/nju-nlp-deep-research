# 华为云运行手册

这份文档用于你把代码 `git push` 到自己的 GitHub fork 之后，在老师提供的华为云服务器上复现实验、生成 `submission.jsonl` 和评估结果。

下面命令默认在项目根目录执行。

## 1. 拉取你的仓库

```bash
git clone <你的 GitHub 仓库 URL>
cd nju-nlp-deep-research
```

如果云服务器上已经 clone 过：

```bash
cd nju-nlp-deep-research
git pull
```

## 2. 准备 Python 环境

需要使用 Python 3.10 或更高版本。课程镜像如果已经有可用环境，可以直接进入该环境。否则新建一个虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
pip install -r agent/requirements.txt
```

如果运行 `vllm` 命令提示不存在，优先使用课程平台预装环境或助教给的安装方式；不要随意升级系统 CUDA/Ascend 相关组件。

## 3. 准备模型

推荐先用 Qwen3-8B：

```bash
git clone https://atomgit.com/hf_mirrors/MindSpore-Lab/Qwen3-8B.git
```

备选 Pangu / DeepDiver：

```bash
git clone https://atomgit.com/ascend-tribe/openPangu-Embedded-7B-DeepDiver.git
```

## 4. 启动 vLLM 服务

建议开一个独立终端或 `tmux` 窗口，让服务一直运行。

Qwen 路线：

```bash
vllm serve ./Qwen3-8B \
  --served-model-name qwen_auto \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
```

Pangu 路线：

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

下面的 agent 命令默认使用：

```text
http://127.0.0.1:8000/v1
```

## 5. 构建 BM25 索引

索引只需要构建一次：

```bash
python -m agent.build_bm25_index \
  --corpus-path ./browsecomp-plus-corpus \
  --index-path ./indexes/browsecomp_plus_bm25.sqlite \
  --overwrite
```

后续重复实验不用再加 `--overwrite`，直接复用 `indexes/browsecomp_plus_bm25.sqlite`。

## 6. 先小规模调试

另开一个终端，确认 vLLM 服务还在运行，然后先跑 3 条：

```bash
python -m agent.deep_research_agent \
  --dataset browsecomp_plus_hard50.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --output runs/debug_submission.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --limit 3 \
  --top-k 8 \
  --max-rounds 6
```

如果用 Pangu，把 `--model qwen_auto` 改成：

```bash
--model pangu_auto
```

## 7. 生成正式 submission

```bash
python -m agent.deep_research_agent \
  --dataset browsecomp_plus_hard50.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --output runs/submission.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --top-k 8 \
  --max-rounds 6 \
  --max-tokens 1024
```

如果运行中断，用下面命令续跑，会跳过已经写入 `runs/submission.jsonl` 的 `query_id`：

```bash
python -m agent.deep_research_agent \
  --dataset browsecomp_plus_hard50.jsonl \
  --index-path indexes/browsecomp_plus_bm25.sqlite \
  --output runs/submission.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --top-k 8 \
  --max-rounds 6 \
  --max-tokens 1024 \
  --resume
```

## 8. 自动评估

```bash
python -m agent.eval \
  --submission runs/submission.jsonl \
  --dataset browsecomp_plus_hard50.jsonl \
  --model qwen_auto \
  --base-url http://127.0.0.1:8000/v1 \
  --output runs/eval_results.jsonl
```

评估结束后，第一行 summary 里有 `accuracy`。也可以看命令行输出的 `Accuracy`。

## 9. 提交文件整理

按老师模板整理目录，示例：

```text
学号-姓名-acc=10_5/
├── 学号-姓名-acc=10_5.pdf
├── core/
│   └── agent/
├── eval/
│   ├── 学号-姓名-submission-10_5.jsonl
│   └── eval.txt
└── README.md
```

其中：

- `runs/submission.jsonl` 改名为 `学号-姓名-submission-最终得分.jsonl`
- 小数点用下划线，例如 `10.5` 写成 `10_5`
- `eval.txt` 可以保存评估命令的终端输出和 summary
- 不做 Open Track 时，目录名不要带 `-opentrack`

## 10. 常见问题

`BM25 index not found`：先执行第 5 步构建索引，或检查 `--index-path` 是否写错。

`Connection refused`：vLLM 服务没启动、端口不是 8000，或 `--base-url` 写错。

模型一直不调用工具：确认启动 vLLM 时带了 `--enable-auto-tool-choice` 和对应 `--tool-call-parser`。脚本默认会先做一次确定性的 bootstrap search，至少能保证每题有初始检索轨迹。

结果中断：用第 7 步的 `--resume` 继续。
