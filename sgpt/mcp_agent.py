import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from click import UsageError

from sgpt.handlers.handler import (
    additional_kwargs as handler_additional_kwargs,
    completion as llm_completion,
    use_litellm,
)

from .mcp import FastMCPExecutor


@dataclass
class MCPActionPlan:
    tool: str
    arguments: Dict[str, Any]
    rationale: str


@dataclass
class MCPPlan:
    preamble: str
    actions: List[MCPActionPlan]
    should_continue: bool


@dataclass
class MCPActionResult:
    plan: MCPActionPlan
    output: str
    elapsed: float
    raw: Any


class MCPAgent:
    """Coordinates planning, MCP tool execution, and summarisation."""

    def __init__(self, executor: FastMCPExecutor):
        self.executor = executor
        self._tool_cache: List[Dict[str, Any]] = []

    # Public API ------------------------------------------------------------------
    def plan(
        self,
        *,
        prompt: str,
        history: List[Dict[str, Any]],
        max_iterations: int,
        model: str,
        temperature: float,
        top_p: float,
    ) -> MCPPlan:
        tools = self._load_tools()
        tool_index = {tool.get("name"): tool for tool in tools if tool.get("name")}
        catalog = json.dumps(tools, ensure_ascii=False, indent=2)
        history_json = json.dumps(history, ensure_ascii=False, indent=2)

        system_message = (
            "你是 ShellGPT 的 MCP 调度器。你会拿到用户请求、历史执行记录以及可用 MCP 工具列表。\n"
            f"一次对话最多执行 {max_iterations} 轮工具调用，如果无法再产生新信息，要立即停止，并将 continue 设为 false。\n"
            "如果连续两次得到的工具输出内容高度相似，也要停止继续调用。始终回复严格的 JSON，禁止添加额外文本或解释。\n"
            "JSON 结构：\n"
            "{\n"
            '  "preamble": "向用户说明你接下来要做什么的简短话术，可为空字符串",\n'
            '  "actions": [\n'
            '    {\n'
            '      "tool": "工具名称，必须来自工具列表",\n'
            '      "arguments": { 任意 JSON 对象，作为工具参数 },\n'
            '      "rationale": "选择该工具的原因"\n'
            "    }\n"
            "  ]\n"
            '  "continue": true 或 false （true 表示执行完这些工具后还需要继续规划，false 表示可以直接总结）\n'
            "}\n"
            "当不需要工具时，让 actions 为 []，同时让 preamble 明确告诉用户你会直接回答。\n"
            "工具列表：\n"
            f"{catalog}\n"
            "历史执行记录：\n"
            f"{history_json if history else '[]'}\n"
            "输出必须是单个 JSON 对象，不能使用代码块。"
        )

        payload = {
            "prompt": prompt,
            "history": history,
            "max_iterations": max_iterations,
        }
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        response_text = self._call_llm(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )
        plan_payload = self._parse_json_object(response_text, "MCP plan")

        preamble = str(plan_payload.get("preamble", "")).strip()
        raw_actions = plan_payload.get("actions", [])
        if raw_actions is None:
            raw_actions = []
        if not isinstance(raw_actions, list):
            raise UsageError("LLM MCP 计划返回的 actions 不是数组。")

        actions: List[MCPActionPlan] = []
        for raw in raw_actions:
            if not isinstance(raw, dict):
                raise UsageError("LLM MCP 计划中存在非对象的 action。")
            tool_name = str(raw.get("tool", "")).strip()
            if not tool_name:
                raise UsageError("LLM MCP 计划缺少工具名称。")
            if tool_name not in tool_index:
                raise UsageError(
                    f'LLM MCP 计划引用了未知工具 "{tool_name}"。'
                )
            arguments = raw.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise UsageError(
                    f'MCP 工具 "{tool_name}" 的参数必须是 JSON 对象。'
                )
            rationale = str(raw.get("rationale", "")).strip()
            actions.append(
                MCPActionPlan(
                    tool=tool_name,
                    arguments=arguments,
                    rationale=rationale,
                )
            )

        should_continue = plan_payload.get("continue")
        if should_continue is None:
            should_continue = plan_payload.get("continue_planning")
        if should_continue is None:
            should_continue = plan_payload.get("has_more")
        should_continue_bool = (
            bool(should_continue)
            if should_continue is not None
            else bool(actions)
        )

        return MCPPlan(
            preamble=preamble,
            actions=actions,
            should_continue=should_continue_bool,
        )

    def execute_actions(self, plan: MCPPlan) -> Tuple[List[MCPActionResult], float]:
        if not plan.actions:
            return [], 0.0

        results: List[MCPActionResult] = []
        total_elapsed = 0.0
        for action in plan.actions:
            raw_result, elapsed = self.executor.call_tool(
                action.tool, action.arguments
            )
            total_elapsed += elapsed
            if self.executor.is_error(raw_result):
                message = self.executor.format_tool_output(raw_result)
                raise UsageError(
                    f'MCP 工具 "{action.tool}" 执行失败: {message}'
                )
            output = self.executor.format_tool_output(raw_result)
            results.append(
                MCPActionResult(
                    plan=action,
                    output=output,
                    elapsed=elapsed,
                    raw=raw_result,
                )
            )

        return results, total_elapsed

    def summarize(
        self,
        *,
        prompt: str,
        preambles: List[str],
        history: List[Dict[str, Any]],
        results: List[MCPActionResult],
        limit_reached: bool,
        model: str,
        temperature: float,
        top_p: float,
    ) -> str:
        payload = {
            "user_prompt": prompt,
            "preambles": preambles,
            "history": history,
            "actions": [
                {
                    "tool": result.plan.tool,
                    "arguments": result.plan.arguments,
                    "rationale": result.plan.rationale,
                    "output": result.output,
                    "elapsed": round(result.elapsed, 3),
                }
                for result in results
            ],
            "limit_reached": limit_reached,
        }

        system_message = (
            "你是 ShellGPT 的答复助手。你将收到一个 JSON，其中包含：\n"
            "- user_prompt: 用户的原始请求；\n"
            "- preambles: 已经告知用户的提示列表；\n"
            "- history: 每一步 MCP 调用的执行记录；\n"
            "- actions: 最新一批 MCP 调用的详细输出（可能为空列表）。\n"
            "- limit_reached: 是否已经达到最大尝试次数（true/false）。\n"
            "请基于这些信息生成最终回复：\n"
            "1. 回答语言要和用户请求一致。\n"
            "2. 如果 actions 非空，要自然地提及使用了哪些工具，并综合输出给出结果。\n"
            "3. 如果 actions 为空，直接回答问题，不要提 MCP。\n"
            "4. 输出纯文本，不要使用 JSON 或代码块。\n"
            "5. 用户已经看到 preambles 中的提示，回答时注意衔接但不要逐字复述。\n"
            "6. 如果 limit_reached 为 true，要如实说明已经达到尝试上限，并基于现有信息给出最佳回答。"
        )

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        summary_text = self._call_llm(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )
        return summary_text.strip()

    # Internal helpers ------------------------------------------------------------
    def _load_tools(self) -> List[Dict[str, Any]]:
        if not self._tool_cache:
            try:
                self._tool_cache = self.executor.list_tools()
            except UsageError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                raise UsageError(f"无法列出 MCP 工具: {exc}") from exc
        return self._tool_cache

    def _call_llm(
        self,
        *,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        top_p: float,
    ) -> str:
        kwargs = dict(handler_additional_kwargs)
        # Remove tool-call leftovers that the main handler might have set.
        kwargs.pop("tool_choice", None)
        kwargs.pop("tools", None)
        kwargs.pop("parallel_tool_calls", None)

        response = llm_completion(
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=messages,
            stream=False,
            **kwargs,
        )
        if use_litellm:
            content = response["choices"][0]["message"]["content"]  # type: ignore[index]
        else:
            choice = response.choices[0]
            content = choice.message.content if choice.message else None  # type: ignore[attr-defined]
        if not content:
            raise UsageError("LLM 返回了空响应，无法继续 MCP 流程。")
        return content

    def _parse_json_object(self, text: str, context: str) -> Dict[str, Any]:
        cleaned = text.strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise UsageError(f"{context} 未返回有效的 JSON。")
        json_candidate = cleaned[start : end + 1]
        try:
            parsed = json.loads(json_candidate)
        except json.JSONDecodeError as exc:
            raise UsageError(f"{context} JSON 解析失败: {exc}") from exc
        if not isinstance(parsed, dict):
            raise UsageError(f"{context} JSON 不是对象。")
        return parsed
