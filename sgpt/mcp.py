import asyncio
import copy
import inspect
import json
from collections.abc import AsyncIterable
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from click import UsageError


class MCPConfig:
    """Reads and validates MCP configuration files."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self._cache: Optional[Dict[str, Dict[str, Any]]] = None

    def load(self) -> Dict[str, Dict[str, Any]]:
        if self._cache is not None:
            return self._cache

        if not self.config_path.exists():
            raise UsageError(
                f"MCP servers config not found at {self.config_path}. "
                "Create it with the desired servers, for example:\n"
                '{\n  "mcpServers": {\n    "default": {"type": "sse", "url": "http://localhost:3000"}\n  }\n}'
            )

        try:
            raw_config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise UsageError(
                f"Invalid MCP servers config at {self.config_path}: {exc}"
            ) from exc

        servers = raw_config.get("mcpServers")
        if not isinstance(servers, dict) or not servers:
            raise UsageError(
                f"No MCP servers defined in {self.config_path}. "
                'Expected JSON structure with top-level key "mcpServers".'
            )

        self._cache = servers
        return servers

    def get_server(self, name: Optional[str]) -> tuple[str, Dict[str, Any]]:
        servers = self.load()
        if name:
            if name not in servers:
                raise UsageError(
                    f'Unknown MCP server "{name}". Available servers: {", ".join(servers)}'
                )
            return name, servers[name]

        if "default" in servers:
            return "default", servers["default"]

        first_name = next(iter(servers))
        return first_name, servers[first_name]

    @staticmethod
    def describe(server_config: Dict[str, Any]) -> str:
        if "url" in server_config:
            return str(server_config["url"])

        bits: Iterable[str] = (
            str(server_config.get("command", "")),
            " ".join(server_config.get("args", [])),
        )
        description = " ".join(filter(None, bits)).strip()
        return description or "<unspecified>"


class FastMCPExecutor:
    """Thin wrapper around fastmcp client functionality."""

    def __init__(
        self,
        server_name: str,
        server_config: Dict[str, Any],
        *,
        api_key: str,
        base_url: Optional[str],
        model: str,
        client_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.server_name = server_name
        self.server_config = server_config
        self.api_key = api_key
        self.base_url = base_url if base_url and base_url.lower() != "default" else None
        self.model = model
        self._client_factory = client_factory
        self._client: Optional[Any] = None

    # Public API -----------------------------------------------------------------
    # Lazily created fastmcp client ------------------------------------------------
    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = self._create_client()
        return self._client

    # Discovery -------------------------------------------------------------------
    def list_tools(self) -> List[Dict[str, Any]]:
        try:
            tools = asyncio.run(self._list_tools_async())
        except AttributeError as exc:  # pragma: no cover - mismatched fastmcp versions
            raise UsageError(
                "fastmcp client does not expose list_tools(). Upgrade fastmcp."
            ) from exc
        return [self._object_to_dict(tool) for tool in tools]

    # Execution -------------------------------------------------------------------
    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]]) -> Tuple[Any, float]:
        safe_arguments = arguments or {}
        if not isinstance(safe_arguments, dict):
            raise UsageError(
                "Tool arguments must be a JSON object. "
                f"Received {type(arguments).__name__} instead."
            )
        try:
            start = perf_counter()
            result = asyncio.run(self._call_tool_async(name, safe_arguments))
            elapsed = perf_counter() - start
        except AttributeError as exc:  # pragma: no cover - mismatched fastmcp versions
            raise UsageError(
                "fastmcp client does not expose call_tool(). Upgrade fastmcp."
            ) from exc
        return result, elapsed

    # Formatting ------------------------------------------------------------------
    @staticmethod
    def is_error(result: Any) -> bool:
        if hasattr(result, "is_error"):
            try:
                return bool(getattr(result, "is_error"))
            except Exception:  # pragma: no cover - defensive
                return False
        if isinstance(result, dict):
            return bool(result.get("is_error"))
        return False

    def format_tool_output(self, result: Any) -> str:
        if result is None:
            return ""

        content = None
        if isinstance(result, dict):
            content = result.get("content")
        elif hasattr(result, "content"):
            content = getattr(result, "content")

        if content is not None:
            rendered = self._render_content(content)
            if rendered:
                return rendered

        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False)

        if hasattr(result, "model_dump"):
            try:
                return json.dumps(result.model_dump(), ensure_ascii=False)
            except Exception:  # pragma: no cover - defensive
                pass

        return str(result)

    async def _list_tools_async(self) -> List[Any]:
        async def operation(client: Any) -> List[Any]:
            raw_result = await client.list_tools()
            return await self._materialize_tools(raw_result)

        return await self._with_client(operation)

    async def _call_tool_async(self, name: str, arguments: Dict[str, Any]) -> Any:
        async def operation(client: Any) -> Any:
            return await client.call_tool(name, arguments)

        return await self._with_client(operation)

    async def _materialize_tools(self, payload: Any) -> List[Any]:
        candidate = payload
        if hasattr(payload, "tools"):
            candidate = payload.tools
            if callable(candidate):
                candidate = candidate()

        if inspect.isawaitable(candidate):
            candidate = await candidate

        if isinstance(candidate, AsyncIterable):
            items: List[Any] = []
            async for item in candidate:
                items.append(item)
            return items

        if isinstance(candidate, list):
            return candidate

        if isinstance(candidate, dict):
            return list(candidate.values())

        if isinstance(candidate, Iterable):
            return list(candidate)

        raise UsageError(
            f"Unexpected result from MCP list_tools(): {type(payload).__name__}"
        )

    async def _with_client(self, operation: Callable[[Any], Any]) -> Any:
        base_client = self.client
        working_client = base_client

        try:
            new_method = getattr(base_client, "new", None)
            if callable(new_method):
                candidate = new_method()
                if candidate is not None:
                    working_client = candidate
        except Exception:  # pragma: no cover - defensive
            working_client = base_client

        if hasattr(working_client, "__aenter__"):
            async with working_client as connected:
                target = connected if connected is not None else working_client
                return await operation(target)

        connect = getattr(working_client, "connect", None)
        close = getattr(working_client, "close", None)

        if callable(connect):
            maybe = connect()
            if inspect.isawaitable(maybe):
                await maybe

        try:
            return await operation(working_client)
        finally:
            if callable(close):
                maybe_close = close()
                if inspect.isawaitable(maybe_close):
                    try:
                        await maybe_close
                    except Exception:
                        pass

    # Internal helpers -----------------------------------------------------------
    def _create_client(self) -> Any:
        if self._client_factory:
            return self._client_factory(
                server_name=self.server_name,
                server_config=self.server_config,
                api_key=self.api_key,
                base_url=self.base_url,
                model=self.model,
            )

        fastmcp_root = self._import_fastmcp()
        attempts: List[Tuple[str, Callable[[], Any]]] = []
        errors: List[str] = []

        client_classes: List[Tuple[str, Any]] = []
        if hasattr(fastmcp_root, "Client"):
            client_classes.append(("fastmcp.Client", fastmcp_root.Client))
        if hasattr(fastmcp_root, "client") and hasattr(fastmcp_root.client, "Client"):
            client_classes.append(
                ("fastmcp.client.Client", fastmcp_root.client.Client)
            )

        if not client_classes:
            raise UsageError(
                "fastmcp.Client not found. Upgrade fastmcp to a recent version."
            )

        configs = [
            self.server_config,
            {"transport": self.server_config},
            {"name": self.server_name, "transport": self.server_config},
        ]

        for origin, cls in client_classes:
            for config in configs:
                config_variant = copy.deepcopy(config)
                attempts.append(
                    (
                        f"{origin}({config_variant})",
                        lambda cls=cls, cfg=config_variant: cls(cfg, name=self.server_name),
                    )
                )

        for description, attempt in attempts:
            try:
                return attempt()
            except Exception as exc:  # pragma: no cover - depends on fastmcp version
                errors.append(f"{description}: {exc}")

        raise UsageError(
            "Unable to initialise fastmcp client. "
            "Ensure fastmcp is installed and compatible.\n"
            + "\n".join(errors)
        )

    def _import_fastmcp(self) -> Any:
        try:
            import fastmcp  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise UsageError(
                "fastmcp is required for --mcp. Install it with 'pip install fastmcp'."
            ) from exc
        return fastmcp

    def _render_content(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content

        if hasattr(content, "model_dump"):
            try:
                return json.dumps(content.model_dump(), ensure_ascii=False)
            except Exception:  # pragma: no cover - defensive
                pass

        if isinstance(content, dict):
            if "type" in content and content.get("type") == "text":
                return str(content.get("text", ""))
            return json.dumps(content, ensure_ascii=False)

        if isinstance(content, Iterable):
            parts: List[str] = []
            for item in content:
                if item is None:
                    continue
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "text":
                        parts.append(str(item.get("text", "")))
                    elif item_type == "image":
                        uri = item.get("uri") or item.get("data")
                        parts.append(f"[image: {uri}]")
                    else:
                        parts.append(json.dumps(item, ensure_ascii=False))
                    continue
                if hasattr(item, "model_dump"):
                    try:
                        parts.append(json.dumps(item.model_dump(), ensure_ascii=False))
                        continue
                    except Exception:  # pragma: no cover - defensive
                        pass
                text_attr = getattr(item, "text", None)
                if text_attr:
                    parts.append(str(text_attr))
                    continue
                parts.append(str(item))
            return "\n".join(part for part in parts if part)

        return str(content)

    def _object_to_dict(self, obj: Any) -> Dict[str, Any]:
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover - defensive
                pass
        result: Dict[str, Any] = {}
        for key in ("name", "description", "input_schema", "inputSchema"):
            if hasattr(obj, key):
                value = getattr(obj, key)
                if value is not None:
                    if key == "inputSchema":
                        result["input_schema"] = value
                    else:
                        result[key] = value
        if not result and hasattr(obj, "__dict__"):
            for key, value in obj.__dict__.items():
                if not key.startswith("_"):
                    result[key] = value
        return result
