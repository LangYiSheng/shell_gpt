import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from click import UsageError

from ..config import cfg
from ..mcp import MCPConfig, FastMCPExecutor
from ..mcp_agent import MCPActionResult, MCPAgent
from ..printer import MarkdownPrinter, TextPrinter


class MCPHandler:
    """Handles --mcp requests by delegating to fastmcp."""

    def __init__(self, markdown: bool, server_name: Optional[str]) -> None:
        self.markdown = markdown
        self.server_name = server_name
        self.config_path = cfg.get("MCP_SERVERS_PATH")
        self.config = MCPConfig(Path(self.config_path))

    @property
    def _printer(self):
        code_theme = cfg.get("CODE_THEME")
        color = cfg.get("DEFAULT_COLOR")
        return MarkdownPrinter(code_theme) if self.markdown else TextPrinter(color)

    def handle(
        self,
        prompt: str,
        *,
        model: str,
        temperature: float,
        top_p: float,
        **_: object,
    ) -> str:
        server_name, server_config = self.config.get_server(self.server_name)
        executor = self._build_executor(server_name, server_config, model)
        agent = MCPAgent(executor)

        history: List[Dict[str, Any]] = []
        preambles: List[str] = []
        aggregated_results: List[MCPActionResult] = []
        descriptor = self.config.describe(server_config)
        max_iterations = 5
        limit_reached = False

        for iteration in range(max_iterations):
            plan = agent.plan(
                prompt=prompt,
                history=history,
                max_iterations=max_iterations,
                model=model,
                temperature=temperature,
                top_p=top_p,
            )

            if plan.preamble and plan.preamble not in preambles:
                preambles.append(plan.preamble)
                typer.echo(plan.preamble)

            if plan.actions:
                typer.echo(
                    f"[ShellGPT:MCP] Plan #{iteration + 1}: "
                    f"{len(plan.actions)} action(s), continue={plan.should_continue}"
                )
                for index, action in enumerate(plan.actions, start=1):
                    typer.echo(
                        f"  • Action {index}: tool='{action.tool}', "
                        f"args={json.dumps(action.arguments, ensure_ascii=False)}"
                    )
                    if action.rationale:
                        typer.echo(f"    rationale: {action.rationale}")

                typer.echo(
                    f"[ShellGPT:MCP] Using server \"{server_name}\" ({descriptor})"
                )

                try:
                    batch_results, batch_elapsed = agent.execute_actions(plan)
                except UsageError:
                    raise
                except Exception as exc:  # pragma: no cover - safeguard
                    raise UsageError(f"fastmcp execution failed: {exc}") from exc

                aggregated_results.extend(batch_results)

                for result in batch_results:
                    history.append(
                        {
                            "tool": result.plan.tool,
                            "arguments": result.plan.arguments,
                            "output": result.output,
                            "elapsed": round(result.elapsed, 3),
                        }
                    )

                typer.echo(f"✔ Response received ({batch_elapsed:.2f}s)")
                for index, result in enumerate(batch_results, start=1):
                    typer.echo(
                        f"  • Result {index}: "
                        f"{result.output if result.output else '[empty output]'}"
                    )
            else:
                if plan.should_continue:
                    raise UsageError(
                        "LLM 决定继续执行 MCP，但未提供任何操作。"
                    )

            if not plan.should_continue:
                break
        else:
            limit_reached = True

        final_message = agent.summarize(
            prompt=prompt,
            preambles=preambles,
            history=history,
            results=aggregated_results,
            limit_reached=limit_reached,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )

        disable_stream = cfg.get("DISABLE_STREAMING") == "true"
        printer = self._printer
        output = printer((chunk for chunk in [final_message]), live=not disable_stream)
        return output

    def _build_executor(
        self,
        server_name: str,
        server_config: Dict[str, Any],
        model: str,
    ) -> FastMCPExecutor:
        api_key = cfg.get("OPENAI_API_KEY")
        base_url = cfg.get("API_BASE_URL")
        return FastMCPExecutor(
            server_name,
            server_config,
            api_key=api_key,
            base_url=base_url,
            model=model,
        )
