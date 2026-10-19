"""MCP tool loader for LangGraph integrations."""

from __future__ import annotations

from pathlib import Path


def default_file_ops_server_config() -> dict[str, dict[str, object]]:
    return {
        "file_ops": {
            "command": "python",
            "args": ["-m", "chat_software.tools.mcp_server"],
            "transport": "stdio",
        }
    }


async def load_mcp_tools(config: dict[str, dict[str, object]] | None = None) -> list[object]:
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:
        raise RuntimeError("Install langchain-mcp-adapters to load MCP tools.") from exc

    client = MultiServerMCPClient(config or default_file_ops_server_config())
    return await client.get_tools()


def reference_filesystem_server_path(reference_root: str | Path) -> Path:
    return Path(reference_root) / "mcp-servers" / "src" / "filesystem"
