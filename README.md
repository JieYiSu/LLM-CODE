# LLM-CODE

**LLM-guided Context-aware Operator Design and Evolution**

LLM-CODE is a tensorized, three-island differential-evolution implementation for
expensive single-objective black-box optimization. It uses a large language model
as an event-triggered operator-structure proposer. Every proposal is parsed into a
typed abstract syntax tree (AST), checked against a bounded operator language, and
accepted only after real-function-evaluation (FE) validation. The search itself
remains tensorized and can run on a CUDA device.

## Features

- event-triggered LLM operator synthesis rather than fixed operator selection only;
- retrieval-augmented operator memory with JSON persistence;
- typed AST validation and bounded vector-expression compilation;
- GPU batch trial generation and three role-specialized islands;
- shared mating, migration, incumbent rollback, and real-FE A/B promotion;
- fixed real-evaluation budgets with checkpointed interruption and resume;
- analytic benchmark, ablation, and current-aware underwater path-planning runners.

## Repository layout

~~~text
llm_operator_de/             Core operator language, compiler, RAG, and islands
evox/                        Bundled EvoX tensorized optimization backend
experiments/                 Benchmark, ablation, and underwater runners
tests/                       Public regression tests
run_expensive_suite.py       Single-objective benchmark entry point
run_all_experiments.py       Code-configured comparison/ablation pipeline
config.example.yaml          Safe provider configuration template
~~~

Generated result files, checkpoints, provider responses, and local configuration
files are intentionally excluded from version control.

## Installation

Python 3.10+ and PyTorch are required. CUDA is recommended for tensorized search.
Install the dependencies required by the selected runner, for example:

~~~powershell
python -m pip install numpy pyyaml openpyxl scipy matplotlib
# Install a PyTorch build appropriate for your CUDA or CPU environment.
~~~

The bundled EvoX source is loaded from evox/src; no separate checkout is needed.

## Local LLM configuration

No API key is included in this repository.

~~~powershell
Copy-Item config.example.yaml config.yaml
~~~

Edit the ignored config.yaml locally:

~~~yaml
openai:
  url: "https://api.example.com/v1"
  api_key: "YOUR_LOCAL_KEY"
  model: "YOUR_MODEL"
~~~

The endpoint must provide an OpenAI-compatible chat-completions interface. Leave
api_key or model empty to disable network calls; the optimizer then continues
with its validated built-in operators. 

## Run a small smoke test

Run commands from the repository root:

~~~powershell
$env:PYTHONPATH = (Resolve-Path .).Path
python -m unittest discover -s tests -v
~~~

For a short benchmark pilot, edit the problem list, dimensions, and repeat count
in run_expensive_suite.py, then run:

~~~powershell
$env:PYTHONPATH = (Resolve-Path .).Path
python run_expensive_suite.py
~~~

The formal protocol is configured in Python. The primary fairness axis is the
number of real objective evaluations; LLM calls and surrogate screening do not
increase the real-FE budget.

## Resume behavior

The suite writes completed combinations to a JSONL progress file and stores
checkpoint state under the configured experiment output directory. Press Ctrl+C
at a safe boundary and run the same entry point again to skip completed
algorithm--problem--dimension--seed combinations. Changed protocols should use a
new output directory.

## LLM operator protocol

The operator semantics and allowed primitives are documented in
llm_operator_de/operator_skill.md and llm_operator_de/operator_catalog.json.
prompts.py constructs the compact request, core.py compiles and validates
the AST, and islands.py coordinates the three search roles. Arbitrary Python
code from a model response is never executed.

## Citation and license

Please cite the accompanying paper when using this implementation. The repository
license is provided in LICENSE. The EvoX subtree retains its own upstream
license and notices.
 
---

# LLM-CODE（中文说明）

LLM-CODE 是面向昂贵单目标黑箱优化的张量化三岛差分进化实现。算法在搜索状态满足触发条件时调用大语言模型提出操作符结构；候选结构会被解析为类型化抽象语法树（AST），经过操作符白名单和复杂度检查，并在真实函数评价（FE）中进行 A/B 验证后才会替换当前操作符。大部分搜索仍由 GPU 上的张量化差分进化完成。

## 主要功能

- 事件触发的大语言模型操作符结构合成；
- 基于 JSON 持久化的检索增强操作符记忆；
- 类型化 AST 校验和受限向量表达式编译；
- GPU 批量试验和探索、平衡、开发三个协作子群；
- 共享交配、迁移、当前操作符回退和真实 FE 晋升；
- 固定真实评价预算、断点保存和中断重启；
- 单目标基准、消融和当前感知水下路径规划实验。

## 目录结构

~~~text
llm_operator_de/             操作符语言、AST 编译器、RAG 和三岛协调器
evox/                        随仓库提供的 EvoX 张量化优化后端
experiments/                 基准、消融和水下工程实验
tests/                       公开回归测试
run_expensive_suite.py       单目标昂贵优化实验入口
run_all_experiments.py       代码内统一配置的实验管线
config.example.yaml          不含密钥的配置模板
~~~

实验结果、检查点、API 响应日志和本地配置默认不会进入 Git。

## 安装与运行

建议使用 Python 3.10 及以上版本，并安装与本机 CUDA 或 CPU 环境匹配的 PyTorch。其余常用依赖可以安装为：

~~~powershell
python -m pip install numpy pyyaml openpyxl scipy matplotlib
~~~

运行测试：

~~~powershell
$env:PYTHONPATH = (Resolve-Path .).Path
python -m unittest discover -s tests -v
~~~

配置大语言模型时，先复制模板：

~~~powershell
Copy-Item config.example.yaml config.yaml
~~~

然后只在本机的 config.yaml 中填写服务地址、模型名和 API 密钥。

## 研究协议

正式实验以真实目标函数评价次数作为公平性标准。LLM 调用、GPU 候选筛选和代理模型
筛选不会增加真实 FE。算法语义和可用原语见：

~~~text
llm_operator_de/operator_skill.md
llm_operator_de/operator_catalog.json
~~~

模型返回的内容只会进入受限 AST 编译器，不会执行任意 Python 代码。

