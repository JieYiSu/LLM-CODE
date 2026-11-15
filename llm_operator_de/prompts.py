"""The explicit LLM operator-synthesis protocol used by LLMOperatorDE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


_PACKAGE_DIR = Path(__file__).resolve().parent


def _load_operator_catalog() -> dict[str, Any]:
    try:
        return json.loads((_PACKAGE_DIR / "operator_catalog.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "rand1": {"formula": "v = r1 + F * (r2 - r3)", "role": "exploration"},
            "best1": {"formula": "v = best + F * (r1 - r2)", "role": "exploitation"},
            "current_to_best1": {"formula": "v = x + F * (best - x) + F * (r1 - r2)", "role": "balance"},
        }


def _load_operator_skill() -> str:
    try:
        return (_PACKAGE_DIR / "operator_skill.md").read_text(encoding="utf-8")
    except OSError:
        return "Use only validated DE primitives, bounded AST nodes, F/CR policies, and fallback operators."


OPERATOR_CATALOG = _load_operator_catalog()
OPERATOR_SKILL = _load_operator_skill()


OPERATOR_SYNTHESIS_SYSTEM_PROMPT = """
你是面向昂贵黑盒优化的差分进化算子程序合成器。

你的任务不是生成最优解，而是根据当前搜索状态生成一个可执行、可验证、
可批量 GPU 执行的 DE 变异、交叉和状态门控程序。

算子表示为 O=(A, theta, pi)：A 是结构，theta 是 F/CR 等参数策略，pi 是
状态门控策略。你只能组合允许的张量原语，不能调用目标函数或修改算法评价流程。

只输出一个 JSON 对象，不要输出解释、Markdown 或思维过程。

以下 operator skill 是算子语义、状态策略和安全边界的规范：
""" + OPERATOR_SKILL


OPERATOR_SYNTHESIS_PROTOCOL = """
【固定接口】
输入：population, fitness, elite_archive, history_archive, state, rng
输出：trial_population，形状必须为 [N, D]

【允许的张量原语】
difference, best_difference, elite_sample, history_sample,
weighted_sum, mask, clip, normalize, constraint_repair

【允许的基础结构】
由运行时加载的 operator_catalog.json 提供；只能使用目录中的 primitive ID。

【可调整内容】
1. 差分向量组合结构；2. F 和 CR 的状态相关策略；
3. 精英档案和历史档案的采样比例；4. 算子混合和状态门控；
5. 探索与开发的切换条件。

【禁止内容】
1. 调用目标函数；2. 访问测试集或未来评价结果；
3. 修改原始目标、约束定义或总函数评价预算（选择、生存者、父代和岛调度只能通过
   algorithm_policy 的验证字段提出）；
4. 使用文件、网络、外部程序或任意 Python 代码执行；
5. 输出无法通过张量形状、边界和复杂度检查的程序。

【输出 JSON schema（推荐格式）】
{
  "operator_id": "唯一字符串",
  "operator_ast": {"op": "primitive|weighted_sum|scale|vector_expr", "...": "..."},
  "operator_code": "可选的 DSL 说明，不是可执行 Python",
  "F": 0.5,
  "CR": 0.9,
  "F_policy": {"stagnation_gain": 0.0, "diversity_gain": 0.0, "budget_gain": 0.0},
  "CR_policy": {"stagnation_gain": 0.0, "diversity_gain": 0.0, "budget_gain": 0.0},
  "gating_policy": {"condition": "always|stagnation_ge|diversity_le", "threshold": 0.0},
  "algorithm_policy": {
    "selection": {"mode": "strict|epsilon|feasibility_first", "epsilon": 0.0},
    "survivor": {"mode": "one_to_one|elitist|diversity_preserving", "elite_fraction": 0.2},
    "constraint": {"mode": "penalty|feasibility_first|epsilon_constraint", "epsilon": 0.0},
    "parent_sampling": {"mode": "uniform_distinct|pbest_biased|archive_mixed", "pbest_fraction": 0.2, "archive_probability": 0.2},
    "island": {"migration_mode": "best_to_worst|round_robin|adaptive", "migration_interval": 5, "role_mode": "preserve|rebalance"}
  },
  "complexity": "简短结构描述",
  "expected_behavior": "简短预期行为",
  "evidence_ids": ["检索案例编号"],
  "fallback_operator": "rand1"
}

最小合法返回只需要 `operator_id` 加上 `operator_ast`（或 `mutation`、`formula`）以及可选的
`F`、`CR`。`operator_code`、`complexity`、`expected_behavior`、`evidence_ids` 和
`algorithm_policy` 都可以省略；不要为了填满 schema 重复解释。

【operator_ast 推荐格式】
primitive 节点必须写成 {"op":"primitive","name":"rand1"}。
加权集成必须写成：
{"op":"weighted_sum","terms":[
  {"op":"primitive","name":"current_to_pbest1","weight":0.6},
  {"op":"primitive","name":"rand1","weight":0.4}
]}
不要使用 children/weights 替代 terms；运行时会校验 AST 结构。

【外部返回格式的兼容性】
推荐格式不是模型必须死记的唯一写法。运行时也接受：
1. 顶层直接返回表达式节点（例如 {"op":"add", ...}）；
2. 使用 expression 或 expr 包裹表达式；
3. 使用 operator 包裹完整算子对象；
4. 使用 id/name、parameters、code 等常见别名。
运行时会先把这些形式规范化为内部 AST，再执行同一套深度、节点数、形状、边界和预算检查。
请只返回一个完整 JSON 对象，不要输出思考过程；结构化请求的答案预算优先用于最终对象。

【创新型 vector_expr】
当已有原语组合不足以表达当前状态时，可以生成新的受限向量表达式：
{
  "op":"vector_expr",
  "expr":{
    "op":"add",
    "left":{"op":"ref","name":"x"},
    "right":{"op":"scale","factor":0.4,"child":{
      "op":"sub",
      "left":{"op":"ref","name":"pbest"},
      "right":{"op":"ref","name":"x"}
    }}
  }
}
可用 ref：x、best、pbest、elite、history、r1、r2、r3、r4；可用操作：
add、sub、scale、weighted_sum、clip、state_scale。只能使用这些操作，不能输出 Python。
"""


def build_operator_synthesis_prompt(state: Mapping[str, Any], rag_context: str) -> str:
    """Build the complete visible prompt sent to the operator LLM."""
    # Compact JSON preserves the strict protocol while removing indentation
    # tokens that do not improve operator synthesis quality.
    state_json = json.dumps(
        dict(state), ensure_ascii=False, default=str, separators=(",", ":")
    )
    catalog_json = json.dumps(
        OPERATOR_CATALOG, ensure_ascii=False, separators=(",", ":")
    )
    return (
        OPERATOR_SYNTHESIS_PROTOCOL
        + "\n【已知算子目录】\n"
        + catalog_json
        + "\n【当前搜索状态】\n"
        + state_json
        + "\n【RAG 检索到的历史算子案例】\n"
        + (rag_context or "[]")
        + "\n如果当前搜索状态中的 innovation_required=true，必须返回根节点为 vector_expr 的新型表达式，"
        + "不要只返回 weighted_sum；否则可根据状态选择已有原语组合。"
        + "\n请基于相似案例生成一个候选算子，并严格返回上述 JSON。"
    )
